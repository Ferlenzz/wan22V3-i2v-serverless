# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Системные пакеты
RUN apt-get update && apt-get install -y \
    python3 python3-pip git git-lfs ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

# Обновляем pip/сборочные тулзы
RUN python3 -m pip install --upgrade pip setuptools wheel

# Библиотеки рантайма
# ПИНЫ под CUDA 12.1 (совместимая связка)
RUN pip3 install --no-cache-dir \
    runpod==1.* \
    "huggingface_hub[cli]==0.24.6" \
 && pip3 install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Код Wan2.2
RUN git clone https://github.com/Wan-Video/Wan2.2.git /workspace/Wan2.2
RUN pip3 install --no-cache-dir -r /workspace/Wan2.2/requirements.txt

# Кэш HF и веса
ENV HF_HOME=/models/.cache/huggingface
RUN mkdir -p $HF_HOME

# Предзагрузка весов (через build-secret HF_TOKEN; если не нужен, секрет можно не задавать)
RUN --mount=type=secret,id=HF_TOKEN bash -lc '\
  set -e; \
  if [ -f /run/secrets/HF_TOKEN ]; then \
      export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
      huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential; \
  fi; \
  huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /models/Wan2.2-TI2V-5B \
'

# Пути для handler.py
ENV WAN_REPO=/workspace/Wan2.2
ENV WAN_CKPT_DIR=/models/Wan2.2-TI2V-5B

# Ваш серверлес-хендлер
WORKDIR /app
COPY handler.py /app/handler.py

# Автопрогрев (0/1)
ENV WARMUP=0

CMD ["python3", "/app/handler.py"]
