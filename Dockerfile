# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1              # оффлайн по умолчанию

# Утилиты
RUN apt-get update && apt-get install -y ffmpeg git git-lfs ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Обновить pip/тулзы
RUN python -m pip install --upgrade pip setuptools wheel

# Лёгкие зависимости (все бинарные колёса — без компиляции)
RUN pip install --no-cache-dir \
    runpod==1.* \
    diffusers==0.30.0 transformers==4.43.3 accelerate==0.33.0 \
    safetensors==0.4.4 imageio[ffmpeg]==2.36.0 pillow==10.4.0 opencv-python==4.10.0.84

# --- Предзагрузка модели (офлайн) через git LFS с ретраями ---
ENV I2V_MODEL_ID=ali-vilab/i2vgen-xl
RUN bash -lc '\
  set -e; \
  mkdir -p /models; \
  for i in 1 2 3; do \
    git lfs install; \
    git lfs clone https://huggingface.co/${I2V_MODEL_ID} /models/i2vgen-xl && break || { \
      echo "git lfs clone failed, retry $i/3"; \
      rm -rf /models/i2vgen-xl; \
      sleep 15; \
    }; \
  done; \
  # убрать историю и LFS-интермедиаты
  rm -rf /models/i2vgen-xl/.git /root/.cache/huggingface || true; \
  du -sh /models/i2vgen-xl || true \
'

# Кэш артефактов (last image / last video)
RUN mkdir -p /cache
ENV CACHE_DIR=/cache \
    MODEL_DIR=/models/i2vgen-xl

WORKDIR /app
COPY handler.py /app/handler.py

# Прогрев при старте (держим модель горячей)
ENV WARMUP=1 \
    CPU_OFFLOAD=0

CMD ["python", "/app/handler.py"]
