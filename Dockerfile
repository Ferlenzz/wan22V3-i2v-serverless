# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

# Системные утилиты
RUN apt-get update && apt-get install -y \
      ffmpeg git git-lfs curl ca-certificates unzip tar \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Python-зависимости (без open-clip / timm / kornia / lightning)
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
      runpod==1.* \
      einops==0.8.0 omegaconf==2.3.0 \
      imageio[ffmpeg]==2.36.0 opencv-python==4.10.0.84 \
      decord==0.6.0 tqdm pyyaml safetensors \
      transformers==4.43.3 accelerate==0.33.0

# Положите распакованный архив AILab-CVC/VideoCrafter в каталог vc1/
# (т.е. в репозитории должен быть путь ./vc1/scripts/run_image2video.sh)
COPY vc1/ /vc1/
# Нормализуем концовки строк и выдадим +x на скрипты
RUN find /vc1 -type f -name '*.sh' -exec sed -i 's/\r$//' {} \; \
 && chmod +x /vc1/scripts/run_image2video.sh || true

# Кэш для временных файлов
ENV CACHE_DIR=/cache
RUN mkdir -p $CACHE_DIR

# Значения по-умолчанию (можно переопределять в Endpoint ENV)
ENV VC1_SCRIPT="/vc1/scripts/run_image2video.sh" \
    VC1_CFG="/vc1/configs/inference_i2v_512_vc1_offline.yaml"

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
