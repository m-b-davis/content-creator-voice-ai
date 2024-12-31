FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including FFmpeg and build essentials
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Upgrade pip and install setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first
RUN pip install --no-cache-dir numpy==1.24.3

# Install PyTorch with CPU support
RUN pip install --no-cache-dir torch==2.2.1+cpu torchaudio==2.2.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Then install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Expose port 8501
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "test/streamlit.py", "--server.address", "0.0.0.0"]