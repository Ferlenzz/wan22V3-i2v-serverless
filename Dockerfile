# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

# Утилиты и системные либы для ffmpeg/opencv
RUN apt-get update && apt-get install -y \
      ffmpeg git git-lfs curl ca-certificates unzip tar \
      findutils bash \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# pip/зависимости
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
      runpod==1.* \
      einops==0.8.0 omegaconf==2.3.0 \
      imageio[ffmpeg]==2.36.0 opencv-python==4.10.0.84 \
      decord==0.6.0 tqdm pyyaml safetensors \
      transformers==4.43.3 accelerate==0.33.0 \
      pytorch-lightning==1.9.5 torchmetrics==0.11.4 \
      kornia==0.7.2 open-clip-torch==2.24.0 timm==0.9.16

# >>> исходники VC (из workflow распакованы в ./vc2) — БЕЗ сетевых скачиваний
COPY vc2/ /vc2/

# Нормализуем CRLF у всех .sh, ищем run_image2video.sh в любом месте /vc2,
# даём +x и создаём удобный симлинк /vc2/run_image2video.sh
RUN set -eux; \
    find /vc2 -type f -name '*.sh' -exec sed -i 's/\r$//' {} \; ; \
    FOUND="$(find /vc2 -type f -iname 'run_image2video.sh' | head -n1 || true)"; \
    if [ -n "$FOUND" ]; then \
        chmod +x "$FOUND"; \
        ln -sf "$FOUND" /vc2/run_image2video.sh; \
    fi

# Кэш и дефолтные ENV (на Runpod можно переопределить)
ENV CACHE_DIR=/cache
RUN mkdir -p $CACHE_DIR

# По умолчанию: используем симлинк, созданный выше
# VC2_CFG можно переопределить через ENV эндпоинта, если у твоей ревизии другой путь
ENV DATA_DIR="" \
    VC2_SCRIPT="/vc2/run_image2video.sh" \
    VC2_CFG="/vc2/configs/inference_i2v_512_v1.0.yaml"

WORKDIR /app
# Текущий обработчик
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
