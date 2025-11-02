# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

# Утилиты
RUN apt-get update && apt-get install -y \
      ffmpeg git git-lfs curl ca-certificates unzip tar && \
    rm -rf /var/lib/apt/lists/* && git lfs install

# Обновить pip/тулзы
RUN python -m pip install --upgrade pip setuptools wheel

# Зависимости для VC/инференса
RUN pip install --no-cache-dir \
      runpod==1.* \
      einops==0.8.0 omegaconf==2.3.0 \
      imageio[ffmpeg]==2.36.0 opencv-python==4.10.0.84 \
      decord==0.6.0 tqdm pyyaml safetensors \
      transformers==4.43.3 accelerate==0.33.0

# >>> исходники VideoCrafter распаковываются в CI в ./vc2
COPY vc2/ /vc2/

# Кэш и дефолтные ENV
ENV CACHE_DIR=/cache
RUN mkdir -p $CACHE_DIR

# Можно переопределять на эндпоинте (если пути в репо отличаются)
ENV DATA_DIR=""
ENV VC2_SCRIPT="/vc2/scripts/inference_i2v.py"
ENV VC2_CFG="/vc2/configs/inference/image2video_512.yaml"

WORKDIR /app
# твой актуальный хендлер
COPY handler_vc2_ckpt.py /app/handler.py

CMD ["python", "/app/handler.py"]
