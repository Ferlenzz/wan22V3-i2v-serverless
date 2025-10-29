# их готовый образ под WAN 2.2 I2V/TI2V
FROM wavespeed/model-deploy:wan22-i2v-latest

RUN apt-get update && apt-get install -y ffmpeg python3-pip git-lfs && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir runpod==1.* "huggingface_hub[cli]"

ENV HF_HOME=/models/.cache/huggingface
RUN mkdir -p $HF_HOME
RUN huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /models/Wan2.2-TI2V-5B

ENV WAN_REPO=/workspace/Wan2.2
ENV WAN_CKPT_DIR=/models/Wan2.2-TI2V-5B

WORKDIR /app
COPY handler.py /app/handler.py

ENV WARMUP=0

CMD ["python3", "/app/handler.py"]
