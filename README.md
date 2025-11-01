# I2VGen-XL — RunPod Serverless (image + prompt)

Простой серверлесс для генерации видео по картинке **и** промпту. Воркеры прогреваются при старте, модель держится в VRAM, есть кэш `last_image`.

## Деплой

1. Соберите образ локально или через GitHub Actions (в этот репозиторий уже добавлен workflow).
2. В RunPod → **Serverless → New Endpoint → Import from Docker Registry**:
   - Image: `docker.io/<YOUR_DH>/i2vgen-xl-serverless:latest`
   - GPU: L4 / 4090 / A5000 (16–24 GB VRAM)
   - **Min Workers: 1** (тёплый воркер)
   - Timeout: 600–900s, Disk: 10–20 GB
   - Env:
     - `WARMUP=1` (по умолчанию уже выставлено в Dockerfile)
     - `CPU_OFFLOAD=1` если VRAM мало (медленнее)

## Вызовы

### 1) Генерация (image + prompt)
```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
 -H "Authorization: Bearer <API_KEY>" -H "Content-Type: application/json" \
 -d '{
  "input":{
    "image":"https://picsum.photos/512",
    "prompt":"cinematic slow push-in, neon reflections, dreamy",
    "num_frames":16,
    "fps":8,
    "width":512,
    "height":512,
    "guidance_scale":7.0
  }
 }'
