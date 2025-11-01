# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=0          

# Утилиты
RUN apt-get update && apt-get install -y \
      ffmpeg git git-lfs ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

# Обновить pip/тулзы
RUN python -m pip install --upgrade pip setuptools wheel

# Зависимости (все бинарные колёса — без компиляции)
RUN pip install --no-cache-dir \
    runpod==1.* \
    diffusers==0.30.0 transformers==4.43.3 accelerate==0.33.0 \
    safetensors==0.4.4 imageio[ffmpeg]==2.36.0 pillow==10.4.0 opencv-python==4.10.0.84

# Каталоги кэша и данных
ENV CACHE_DIR=/cache \
    DATA_DIR=/data \
    MODEL_DIR=/data/models/i2vgen-xl \
    I2V_MODEL_ID=ali-vilab/i2vgen-xl \
    WARMUP=1 \
    CPU_OFFLOAD=0

RUN mkdir -p /cache

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
