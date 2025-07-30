FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY pose_api.py .
COPY model3-yolol/yolo11l-pose.pt ./model3-yolol/

RUN mkdir -p /app/images

EXPOSE 60001

CMD ["uvicorn", "pose_api:app", "--host", "0.0.0.0", "--port", "60001", "--workers", "3"]
