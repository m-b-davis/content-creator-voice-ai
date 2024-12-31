import os
import time
import librosa
import soundfile
import streamlit as st
import torch
from io import BytesIO
import subprocess
import tempfile
from voicefixer import VoiceFixer
import shutil
import logging
import sys
from datetime import datetime
import signal
from contextlib import contextmanager
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_info(message):
    """Log info message and display in Streamlit"""
    logger.info(message)
    st.write(message)

class TimeoutError(Exception):
    pass

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=300):
    """Run a function with a timeout using threading"""
    result = []
    error = []
    
    def worker():
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            error.append(e)
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        thread.join(1)  # Give it a second to clean up
        raise TimeoutError("Operation timed out")
    
    if error:
        raise error[0]
    
    return result[0] if result else None

# Page configuration
st.set_page_config(
    page_title="VoiceBoost AI - Professional Voice Enhancement for Creators",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0 3rem;
        background-color: #0e1117;
        color: white;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #a0a0a0;
        margin-bottom: 2rem;
    }
    
    /* File uploader styling */
    .stFileUploader {
        width: 100%;
    }
    
    .uploadedFile {
        display: none;
    }
    
    .stFileUploader > div {
        background: #1e2127;
        border-radius: 12px;
        padding: 4rem 2rem !important;
        text-align: center;
        border: 2px dashed #2d3139;
        color: white;
    }
    
    .stFileUploader > div:hover {
        border-color: #FFD700;
        background: #262931;
        cursor: pointer;
    }
    
    /* Status indicators */
    .stStatus {
        background-color: #1e2127 !important;
        border: 1px solid #2d3139 !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background-color: #FFD700 !important;
        color: black !important;
        padding: 1.5rem !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin: 2rem 0 !important;
    }
    
    .stDownloadButton > button:hover {
        background-color: #FFC000 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.2) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #FFD700 !important;
    }
    
    /* Toggle switch styling */
    .switch-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .version-label {
        color: #a0a0a0;
        font-size: 1rem;
    }
    
    .version-label.active {
        color: white;
        font-weight: 600;
    }
    
    /* Style the toggle switch */
    .stCheckbox {
        display: flex;
        justify-content: center;
    }
    
    .stCheckbox > div {
        background: transparent !important;
    }
    
    .stCheckbox > div > div > div {
        background-color: #FFD700 !important;
    }
    
    /* Center video title */
    .video-title {
        text-align: center;
        margin: 2rem 0 1rem 0;
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize VoiceFixer with error handling
@st.cache_resource
def init_voicefixer():
    try:
        return VoiceFixer()
    except Exception as e:
        st.error(f"Error initializing VoiceFixer: {str(e)}")
        return None

voice_fixer = init_voicefixer()
sample_rate = 44100

# Hero Section
st.markdown("""
    <div class="hero-section">
        <h1>🎤 VoiceBoost AI</h1>
        <p>Make your videos sound professionally produced in seconds</p>
    </div>
    """, unsafe_allow_html=True)

# File Upload Section
video_file = st.file_uploader(
    "Drop your video here",
    type=["mp4", "mov"],
    key="video_uploader",
    help="Maximum file size: 200MB",
)

def check_ffmpeg_installed():
    """Check if ffmpeg is available in the system."""
    if not shutil.which('ffmpeg'):
        st.error("""
        ❌ FFmpeg is not installed. This is required for video processing.
        
        If you're running this locally:
        - Ubuntu/Debian: `sudo apt-get install ffmpeg`
        - MacOS: `brew install ffmpeg`
        - Windows: Download from https://ffmpeg.org/download.html
        
        If you're using Streamlit Cloud, please ensure packages.txt includes ffmpeg.
        """)
        return False
    return True

if not video_file:
    st.markdown("""
        <div style='text-align: center; color: #a0a0a0; font-size: 0.9rem; margin-top: -1rem;'>
        Drag and drop or click to browse • MP4 or MOV • Up to 200MB
        </div>
        """, unsafe_allow_html=True)
else:
    # Add file size check
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    file_size = len(video_file.getvalue())
    if file_size > MAX_FILE_SIZE:
        st.error(f"File too large. Maximum size is 100 MB. Your file is {file_size / (1024*1024):.1f} MB")
        st.stop()

    if not check_ffmpeg_installed():
        st.stop()

    # Create a progress bar
    progress_bar = st.progress(0)
    
    try:
        # Processing Section
        with st.status("Enhancing your video...", expanded=True) as status:
            start_time = datetime.now()
            log_info("⚡ Starting video enhancement process...")
            log_info(f"Input video size: {len(video_file.getvalue()) / (1024*1024):.2f} MB")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded video
                input_video_path = os.path.join(temp_dir, f"input.{video_file.name.split('.')[-1]}")
                with open(input_video_path, 'wb') as f:
                    f.write(video_file.getvalue())
                progress_bar.progress(10)
                
                # Extract audio
                log_info("🎵 Extracting audio from video...")
                input_audio_path = os.path.join(temp_dir, "input.wav")
                try:
                    result = subprocess.run([
                        'ffmpeg', '-i', input_video_path,
                        '-vn', '-acodec', 'pcm_s16le',
                        '-ar', str(sample_rate), '-ac', '1',
                        input_audio_path
                    ], check=True, capture_output=True, text=True)
                    log_info("✅ Audio extraction complete")
                except subprocess.CalledProcessError as e:
                    log_info(f"❌ FFmpeg error output: {e.stderr}")
                    st.error(f"Error processing video: {e.stderr}")
                    st.stop()
                except Exception as e:
                    log_info(f"❌ Unexpected error: {str(e)}")
                    st.error(f"Unexpected error: {str(e)}")
                    st.stop()
                
                progress_bar.progress(30)
                
                # Process audio
                log_info("✨ Applying AI enhancement...")
                try:
                    audio, _ = librosa.load(input_audio_path, sr=sample_rate, mono=True)
                    log_info(f"Audio loaded successfully. Shape: {audio.shape}")
                    
                    try:
                        # Run enhancement with timeout
                        enhanced_audio = run_with_timeout(
                            voice_fixer.restore_inmem,
                            args=(audio,),
                            kwargs={'mode': 2, 'cuda': False},
                            timeout_duration=300
                        )
                        log_info("✅ AI enhancement complete")
                    except TimeoutError:
                        log_info("❌ Enhancement timed out after 5 minutes")
                        st.error("Enhancement process took too long. Please try with a shorter video.")
                        st.stop()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            log_info("❌ Out of memory error")
                            st.error("Not enough memory. Please try with a shorter video.")
                            st.stop()
                        else:
                            raise e
                    
                except Exception as e:
                    log_info(f"❌ Error during audio enhancement: {str(e)}")
                    st.error(f"Error during audio enhancement: {str(e)}")
                    st.stop()
                
                progress_bar.progress(60)
                
                # Save enhanced audio
                log_info("💾 Saving enhanced audio...")
                output_audio_path = os.path.join(temp_dir, "enhanced.wav")
                try:
                    soundfile.write(output_audio_path, enhanced_audio.T, samplerate=sample_rate)
                    log_info("✅ Enhanced audio saved successfully")
                except Exception as e:
                    log_info(f"❌ Error saving enhanced audio: {str(e)}")
                    st.error(f"Error saving enhanced audio: {str(e)}")
                    st.stop()
                
                progress_bar.progress(80)
                
                # Merge with video
                log_info("🎬 Creating final enhanced video...")
                output_video_path = os.path.join(temp_dir, f"enhanced_{video_file.name}")
                try:
                    result = subprocess.run([
                        'ffmpeg', '-i', input_video_path,
                        '-i', output_audio_path,
                        '-c:v', 'copy', '-c:a', 'aac',
                        '-map', '0:v:0', '-map', '1:a:0',
                        output_video_path
                    ], check=True, capture_output=True, text=True)
                    log_info("✅ Video merging complete")
                except subprocess.CalledProcessError as e:
                    log_info(f"❌ Error merging video: {e.stderr}")
                    st.error(f"Error merging video: {e.stderr}")
                    st.stop()
                
                progress_bar.progress(100)
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                log_info(f"✨ Total processing time: {processing_time:.2f} seconds")
                
                status.update(label="Enhancement complete! ✨", state="complete")

                # Preview Section
                st.markdown("<h3 class='video-title'>Preview Comparison</h3>", unsafe_allow_html=True)
                
                # Create two columns for side-by-side comparison
                left_col, right_col = st.columns(2)
                
                with left_col:
                    st.markdown("<p style='text-align: center; color: #a0a0a0;'>Original</p>", unsafe_allow_html=True)
                    st.video(input_video_path)
                    
                with right_col:
                    st.markdown("<p style='text-align: center; color: #FFD700; font-weight: 600;'>Enhanced</p>", unsafe_allow_html=True)
                    st.video(output_video_path)

                # Download Section - centered
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    with open(output_video_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Enhanced Video",
                            data=f.read(),
                            file_name=f"enhanced_{video_file.name}",
                            mime=f"video/{video_file.name.split('.')[-1]}",
                            use_container_width=True,
                        )
    except Exception as e:
        log_info(f"❌ Fatal error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()

# Footer
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #666; font-size: 0.8rem;'>
    🔒 Your videos are processed securely and automatically deleted
    </div>
    """, unsafe_allow_html=True)
