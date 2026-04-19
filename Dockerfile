FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first to avoid pulling the 2GB GPU build
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the YOLO model so it's baked into the image
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"

CMD ["/bin/bash", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
