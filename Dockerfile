# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

# Системные пакеты (ffmpeg/opencv и т.п.)
RUN apt-get update && apt-get install -y \
      ffmpeg git git-lfs curl ca-certificates unzip tar \
      findutils bash \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Python-зависимости (зафиксированы версии для стабильности)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      einops==0.8.0 \
      omegaconf==2.3.0 \
      imageio[ffmpeg]==2.36.0 \
      opencv-python==4.10.0.84 \
      decord==0.6.0 \
      tqdm \
      pyyaml \
      safetensors \
      transformers==4.43.3 \
      accelerate==0.33.0 \
      # зафиксированные — чтобы не приходилось потом добивать руками
      open-clip-torch==2.24.0 \
      timm==0.9.16 \
      kornia==0.7.2 \
      pytorch-lightning==1.9.5 \
      torchmetrics==0.11.4 \
      runpod==1.*

# --- кладём VideoCrafter1 код без сети (zip лежит в репозитории) ---
#   vendor/VideoCrafter-main.zip  -> /vc1
COPY vendor/VideoCrafter-main.zip /tmp/vc1.zip
RUN set -eux; \
    unzip -q /tmp/vc1.zip -d /tmp; \
    mv /tmp/VideoCrafter-main /vc1; \
    # нормализуем CRLF у .sh, выдаём +x, создаём удобный симлинк
    find /vc1 -type f -name '*.sh' -exec sed -i 's/\r$//' {} \; ; \
    if [ -f /vc1/scripts/run_image2video.sh ]; then \
      chmod +x /vc1/scripts/run_image2video.sh; \
      ln -sf /vc1/scripts/run_image2video.sh /vc1/run_image2video.sh; \
    fi

# Наш офлайн-конфиг для VC1 (лежит в репо)
COPY vc1/configs/inference_i2v_512_vc1_offline.yaml /vc1/configs/inference_i2v_512_vc1_offline.yaml

# Кэш и дефолтные ENV (на Runpod переопределишь)
ENV CACHE_DIR=/cache
RUN mkdir -p $CACHE_DIR

# Значения по умолчанию — можно менять в настройках Serverless
ENV DATA_DIR="" \
    VC1_SCRIPT="/vc1/run_image2video.sh" \
    VC1_CFG="/vc1/configs/inference_i2v_512_vc1_offline.yaml"

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
