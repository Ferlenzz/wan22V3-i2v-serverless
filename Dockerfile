# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

# Утилиты
RUN apt-get update && apt-get install -y \
      ffmpeg git git-lfs ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

# Python toolchain
RUN python -m pip install --upgrade pip setuptools wheel

# Зависимости (без компиляции)
RUN pip install --no-cache-dir \
    runpod==1.* \
    diffusers==0.30.0 transformers==4.43.3 accelerate==0.33.0 \
    safetensors==0.4.4 imageio[ffmpeg]==2.36.0 pillow==10.4.0 opencv-python==4.10.0.84

# Кэш в контейнере
RUN mkdir -p /cache

# ВАЖНО: ожидаем модель в Network Volume по пути /data/models/i2vgen-xl
ENV DATA_DIR=/data \
    MODEL_DIR=/data/models/i2vgen-xl \
    CACHE_DIR=/cache \
    WARMUP=1 \
    CPU_OFFLOAD=0

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
