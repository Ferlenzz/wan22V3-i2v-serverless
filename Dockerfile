# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y ffmpeg git git-lfs && \
    rm -rf /var/lib/apt/lists/* && git lfs install

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    runpod==1.* \
    diffusers==0.30.0 transformers==4.43.3 accelerate==0.33.0 \
    safetensors==0.4.4 imageio[ffmpeg]==2.36.0 pillow==10.4.0 opencv-python==4.10.0.84

ENV HF_HOME=/models/.cache/huggingface
RUN mkdir -p $HF_HOME

ENV I2V_MODEL_ID=ali-vilab/i2vgen-xl
RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os
mid=os.environ.get("I2V_MODEL_ID","ali-vilab/i2vgen-xl")
snapshot_download(repo_id=mid, local_dir="/models/i2vgen-xl", local_dir_use_symlinks=False)
PY

RUN mkdir -p /cache
ENV CACHE_DIR=/cache

WORKDIR /app
COPY handler.py /app/handler.py

# ENV:
#  - WARMUP=1 включит автопрогрев при старте контейнера
#  - CPU_OFFLOAD=1 если VRAM мало (медленнее, но экономнее по памяти)
ENV WARMUP=1 \
    CPU_OFFLOAD=0

CMD ["python", "/app/handler.py"]
