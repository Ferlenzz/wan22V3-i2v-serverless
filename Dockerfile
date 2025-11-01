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

# Обновить pip/тулзы
RUN python -m pip install --upgrade pip setuptools wheel

# Зависимости (все бинарные колёса — без компиляции)
RUN pip install --no-cache-dir \
    runpod==1.* \
    diffusers==0.30.0 transformers==4.43.3 accelerate==0.33.0 \
    safetensors==0.4.4 imageio[ffmpeg]==2.36.0 pillow==10.4.0 opencv-python==4.10.0.84

# --- Предзагрузка модели через git + LFS (с ретраями) ---
ENV I2V_MODEL_ID=ali-vilab/i2vgen-xl
RUN bash -lc '\
  set -e; \
  mkdir -p /models; \
  for i in 1 2 3; do \
    echo "Clone attempt $i/3"; \
    rm -rf /models/i2vgen-xl; \
    git clone https://huggingface.co/${I2V_MODEL_ID} /models/i2vgen-xl && \
    cd /models/i2vgen-xl && git lfs pull && cd / && break || { \
      echo "Clone failed, retrying..."; \
      sleep 15; \
    }; \
  done; \
  # почистим историю и кэши
  rm -rf /models/i2vgen-xl/.git /root/.cache/huggingface || true; \
  # контроль: размер и наличие файлов
  du -sh /models/i2vgen-xl || true; \
  test -f /models/i2vgen-xl/model_index.json || (echo "model_index.json missing" && exit 1) \
'

# Кэш артефактов (last image / last video)
RUN mkdir -p /cache
ENV CACHE_DIR=/cache \
    MODEL_DIR=/models/i2vgen-xl \
    WARMUP=1 \
    CPU_OFFLOAD=0

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
