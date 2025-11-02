# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y ffmpeg git git-lfs curl ca-certificates unzip tar && \
    rm -rf /var/lib/apt/lists/* && git lfs install

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    runpod==1.* einops==0.8.0 omegaconf==2.3.0 \
    imageio[ffmpeg]==2.36.0 opencv-python==4.10.0.84 \
    decord==0.6.0 tqdm pyyaml safetensors \
    transformers==4.43.3 accelerate==0.33.0

# исходники VC2 пришли в контекст ./vc2 из шага выше
COPY vc2/ /vc2/

ENV CACHE_DIR=/cache
RUN mkdir -p $CACHE_DIR

ENV DATA_DIR=""
ENV VC2_SCRIPT="/vc2/scripts/inference_i2v.py"
ENV VC2_CFG="/vc2/configs/inference/image2video_512.yaml"

WORKDIR /app
COPY handler_vc2_ckpt.py /app/handler.py
CMD ["python", "/app/handler.py"]
