# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

# Системные пакеты
RUN apt-get update && apt-get install -y \
      ffmpeg git git-lfs curl ca-certificates unzip tar \
      findutils bash \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

# Python-зависимости (добавлены PL/metrics + open-clip/timm/kornia)
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir \
      runpod==1.* \
      einops==0.8.0 omegaconf==2.3.0 \
      imageio[ffmpeg]==2.36.0 opencv-python==4.10.0.84 \
      decord==0.6.0 tqdm pyyaml safetensors \
      transformers==4.43.3 accelerate==0.33.0 \
      pytorch-lightning==1.9.5 torchmetrics==0.11.4 \
      open-clip-torch==2.24.0 timm==0.9.16 kornia==0.7.2

# === VC1: кладём ZIP внутрь образа и распаковываем в /vc1 ===
# (ZIP должен лежать в репозитории: vendor/VideoCrafter-main.zip)
COPY vendor/VideoCrafter-main.zip /tmp/vc1.zip
RUN set -eux; \
    unzip -q /tmp/vc1.zip -d /tmp; \
    mv /tmp/VideoCrafter-main /vc1; \
    # нормализуем CRLF у *.sh
    find /vc1 -type f -name '*.sh' -exec sed -i 's/\r$//' {} \; ; \
    # даём +x и создаём удобный симлинк
    if [ -f /vc1/scripts/run_image2video.sh ]; then \
      chmod +x /vc1/scripts/run_image2video.sh; \
      ln -sf /vc1/scripts/run_image2video.sh /vc1/run_image2video.sh; \
    fi

# Наш офлайн-YAML (лежит в репозитории по пути vc1/configs/)
COPY vc1/configs/inference_i2v_512_vc1_offline.yaml /vc1/configs/inference_i2v_512_vc1_offline.yaml

# Кэш
ENV CACHE_DIR=/cache
RUN mkdir -p $CACHE_DIR

# ==== ENV по умолчанию ====
# Скрипт VC1 и конфиг:
ENV VC1_SCRIPT="/vc1/run_image2video.sh" \
    VC1_CFG="/vc1/configs/inference_i2v_512_vc1_offline.yaml"

# Где искать CLIP (по умолчанию твой путь из скрина):
ENV CLIP_DIR="/runpod-volume/models/transformers/openai-clip-vit-large-patch14"

# Где искать чекпоинт модели; при желании переопредели в Endpoint ENV:
ENV VC1_CKPT="/runpod-volume/checkpoints/i2v_512_v1/model.ckpt"

# DATA_DIR определяешь в Endpoint (обычно /runpod-volume)
ENV DATA_DIR=""

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
