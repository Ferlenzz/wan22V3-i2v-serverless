# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

# Базовые утилиты
RUN apt-get update && apt-get install -y ffmpeg git git-lfs curl ca-certificates unzip tar && \
    rm -rf /var/lib/apt/lists/* && git lfs install

RUN python -m pip install --upgrade pip setuptools wheel

# Библиотеки, которые обычно нужны VC2 I2V-скриптам
RUN pip install --no-cache-dir \
    runpod==1.* \
    einops==0.8.0 omegaconf==2.3.0 \
    imageio[ffmpeg]==2.36.0 opencv-python==4.10.0.84 \
    decord==0.6.0 tqdm pyyaml safetensors \
    transformers==4.43.3 accelerate==0.33.0

# ---- Впечатываем исходники VideoCrafter2 в /vc2 ----
# Используем tarball, чтобы не мучиться с git в рантайме
RUN bash -lc '\
  set -euo pipefail; \
  mkdir -p /vc2; \
  for U in \
    https://codeload.github.com/OpenGVLab/VideoCrafter2/tar.gz/refs/heads/main \
    https://api.github.com/repos/OpenGVLab/VideoCrafter2/tarball \
  ; do \
    echo "Try $U"; \
    rm -f /tmp/vc2.tar.gz; \
    if curl -L --fail -o /tmp/vc2.tar.gz "$U"; then \
      rm -rf /tmp/vc2 && mkdir -p /tmp/vc2; \
      tar -xzf /tmp/vc2.tar.gz -C /tmp/vc2; \
      VC2DIR=$(find /tmp/vc2 -maxdepth 1 -type d -name "*VideoCrafter2*" | head -n1); \
      if [ -n "$VC2DIR" ]; then mv "$VC2DIR"/* /vc2/ && break; fi; \
    fi; \
  done; \
  test -f /vc2/README.md || { echo "VC2 sources not found"; exit 1; }; \
  ls -la /vc2 | head \
'

ENV CACHE_DIR=/cache
RUN mkdir -p $CACHE_DIR

# Автодетект NV: /data | /runpod-volume | /workspace
ENV DATA_DIR=""

# Позволяем переопределять путь к скрипту/конфигу через ENV
ENV VC2_SCRIPT="/vc2/scripts/inference_i2v.py"
ENV VC2_CFG="/vc2/configs/inference/image2video_512.yaml"

WORKDIR /app
COPY handler_vc2_ckpt.py /app/handler.py
CMD ["python", "/app/handler.py"]
