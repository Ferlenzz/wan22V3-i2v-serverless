# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

# Утилиты
RUN apt-get update && apt-get install -y \
      ffmpeg git git-lfs curl ca-certificates unzip tar && \
    rm -rf /var/lib/apt/lists/* && git lfs install

# pip/зависимости
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
      runpod==1.* \
      einops==0.8.0 omegaconf==2.3.0 \
      imageio[ffmpeg]==2.36.0 opencv-python==4.10.0.84 \
      decord==0.6.0 tqdm pyyaml safetensors \
      transformers==4.43.3 accelerate==0.33.0

# >>> исходники VC2 из workflow (./vc2) — НИКАКИХ скачиваний в образе
COPY vc2/ /vc2/
# На некоторых ревизиях .sh может быть без +x — добавим
RUN if [ -f /vc2/run_image2video.sh ]; then chmod +x /vc2/run_image2video.sh; fi

# Кэш и дефолтные ENV (в Runpod переопределишь)
ENV CACHE_DIR=/cache
RUN mkdir -p $CACHE_DIR

ENV DATA_DIR="" \
    VC2_SCRIPT="/vc2/run_image2video.sh" \
    VC2_CFG="/vc2/configs/inference_i2v_512_v1.0.yaml"

WORKDIR /app
# >>> теперь у тебя handler называется именно handler.py
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
