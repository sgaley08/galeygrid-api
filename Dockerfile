FROM python:3.11-slim

# Install system dependencies for OpenCV (required by ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and server
COPY best.pt .
COPY server.py .

# Railway sets PORT env var
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
