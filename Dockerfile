FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git git-lfs ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir runpod==1.* "huggingface_hub[cli]" \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/Wan-Video/Wan2.2.git /workspace/Wan2.2
RUN pip3 install --no-cache-dir -r /workspace/Wan2.2/requirements.txt

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

ENV WAN_REPO=/workspace/Wan2.2
ENV WAN_CKPT_DIR=/models/Wan2.2-TI2V-5B

WORKDIR /app
COPY handler.py /app/handler.py

ENV WARMUP=0

CMD ["python3", "/app/handler.py"]
