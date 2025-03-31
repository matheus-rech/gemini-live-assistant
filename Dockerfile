# Dockerfile for Gemini Voice Reading Assistant

# Stage 1: Build stage
FROM python:3.10-slim as build

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libopencv-dev \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .
COPY setup.sh run.sh ./
RUN chmod +x setup.sh run.sh

# Stage 2: Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed dependencies from build stage
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=build /app /app

# Create logs directory
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV GEMINI_MODEL=models/gemini-2.0-flash-exp

# Expose port for WebSocket server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "server.py"]
