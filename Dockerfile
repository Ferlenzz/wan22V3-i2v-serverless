# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# 1) базовые пакеты
RUN apt-get update && apt-get install -y \
    python3 python3-pip git git-lfs ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

# 2) свежий pip и колёса
RUN python3 -m pip install --upgrade pip setuptools wheel

# 3) python-зависимости (без компиляции flash-attn/xformers)
#    Пины совместимы с CUDA 12.1 (cu121)
RUN pip3 install --no-cache-dir \
    runpod==1.* \
    "huggingface_hub[cli]==0.24.6" \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Минимальный стек вокруг Wan2.2 (без тяжёлых экстеншенов)
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 einops==0.8.0 \
    transformers==4.43.3 accelerate==0.33.0 diffusers==0.30.0 \
    sentencepiece==0.2.0 safetensors==0.4.4 \
    opencv-python==4.10.0.84 pillow==10.4.0 \
    decord==0.6.0 imageio[ffmpeg]==2.36.0 tqdm==4.66.5

# 4) код Wan2.2
RUN git clone https://github.com/Wan-Video/Wan2.2.git /workspace/Wan2.2

# 5) HF cache и предзагрузка весов
ENV HF_HOME=/models/.cache/huggingface
RUN mkdir -p $HF_HOME

RUN --mount=type=secret,id=HF_TOKEN bash -lc '\
  set -e; \
  if [ -f /run/secrets/HF_TOKEN ]; then \
      export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
      huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential; \
  fi; \
  huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /models/Wan2.2-TI2V-5B \
'

# 6) переменные для handler.py
ENV WAN_REPO=/workspace/Wan2.2
ENV WAN_CKPT_DIR=/models/Wan2.2-TI2V-5B

# 7) наш серверлес-хендлер
WORKDIR /app
COPY handler.py /app/handler.py

# опциональный автопрогрев
ENV WARMUP=0

CMD ["python3", "/app/handler.py"]
