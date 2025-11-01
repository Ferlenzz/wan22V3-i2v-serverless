import os, io, base64, tempfile, urllib.request, json
from typing import Optional, List
import imageio.v3 as iio
from PIL import Image

import torch
import runpod
from diffusers import I2VGenXLPipeline, DPMSolverMultistepScheduler

MODEL_DIR = "/models/i2vgen-xl"
MODEL_ID  = os.environ.get("I2V_MODEL_ID", MODEL_DIR)
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", "0") == "1"
CACHE_DIR = os.environ.get("CACHE_DIR", "/cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DTYPE = torch.float16
device = "cuda"

pipe = I2VGenXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    variant="fp16",
    local_files_only=True  
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
if CPU_OFFLOAD:
    pipe.enable_model_cpu_offload()
else:
    pipe.to(device)
pipe.enable_vae_tiling()

LAST_IMAGE_PATH = os.path.join(CACHE_DIR, "last_image.png")
LAST_VIDEO_PATH = os.path.join(CACHE_DIR, "last_video.mp4")

def _pil_from_any(image_field: str) -> Image.Image:
    """Поддержка URL / data:base64 / локальный путь в контейнере."""
    if image_field.startswith("data:"):
        b64 = image_field.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if image_field.startswith("http://") or image_field.startswith("https://"):
        tmp = os.path.join(tempfile.gettempdir(), "i2v.png")
        urllib.request.urlretrieve(image_field, tmp)
        return Image.open(tmp).convert("RGB")
    return Image.open(image_field).convert("RGB")

def _save_last_image(img: Image.Image) -> None:
    img.save(LAST_IMAGE_PATH)

def _load_last_image() -> Optional[Image.Image]:
    if os.path.exists(LAST_IMAGE_PATH):
        try:
            return Image.open(LAST_IMAGE_PATH).convert("RGB")
        except Exception:
            return None
    return None

def _encode_file_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _warmup_once():
    """Одноразовый быстрый прогрев: tiny 8 кадров 256x256 из 1x1 PNG."""
    try:
        px = Image.new("RGB", (1, 1), (255, 255, 255))
        # сохраняем и как last_image, чтобы сразу был use_last_image
        _save_last_image(px.resize((256, 256)))
        _ = pipe(
            prompt="warmup",
            image=px.resize((256, 256)),
            num_frames=8,
            guidance_scale=5.0,
            height=256,
            width=256,
            generator=torch.Generator(device=device).manual_seed(42)
        ).frames
    except Exception as e:
        print("Warmup failed:", e)

if os.environ.get("WARMUP", "0") == "1":
    _warmup_once()

def handler(job):
    """
    Вход (JSON в job['input']):
      - image: URL / data:base64 / путь (если use_last_image=true — можно опустить)
      - prompt: str (опционально)
      - num_frames: int (деф. 16)
      - fps: int (деф. 8)
      - width, height: int (деф. 512, 512)
      - guidance_scale: float (деф. 7.0)
      - seed: int (опц.)
      - use_last_image: bool (деф. false) — использовать сохранённую последнюю картинку
      - store_last: bool (деф. true) — сохранять текущую входную картинку как "последнюю"
      - action: str (опц.) — "get_last_image" | "clear_cache"
    Выход:
      - video_b64
      - (для get_last_image) image_b64
    """
    inp = job.get("input", {}) or {}

    action = inp.get("action")
    if action == "get_last_image":
        img = _load_last_image()
        if not img:
            return {"error": "no last image stored"}
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return {"image_b64": base64.b64encode(bio.getvalue()).decode()}
    if action == "clear_cache":
        status = {"cleared": []}
        for p in [LAST_IMAGE_PATH, LAST_VIDEO_PATH]:
            if os.path.exists(p):
                os.remove(p)
                status["cleared"].append(os.path.basename(p))
        return status

    use_last = bool(inp.get("use_last_image", False))
    store_last = True if inp.get("store_last", True) else False

    image_field = inp.get("image")
    if use_last:
        image = _load_last_image()
        if image is None and image_field:
            image = _pil_from_any(image_field)
        if image is None:
            return {"error": "no last image stored; provide 'image' or set use_last_image=false"}
    else:
        if not image_field:
            return {"error": "provide 'image' (URL / data:base64 / path) or set use_last_image=true"}
        image = _pil_from_any(image_field)

    # Параметры
    prompt = inp.get("prompt", "")
    num_frames = int(inp.get("num_frames", 16))  
    fps = int(inp.get("fps", 8))                 
    width = int(inp.get("width", 512))
    height = int(inp.get("height", 512))
    guidance_scale = float(inp.get("guidance_scale", 7.0))
    seed = inp.get("seed")
    generator = torch.Generator(device=device).manual_seed(int(seed)) if seed else None

    # Генерация
    frames: List[Image.Image] = pipe(
        prompt=prompt,
        image=image,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator
    ).frames

    # Сохраним video и last image
    tmp_mp4 = os.path.join(tempfile.gettempdir(), "out.mp4")
    iio.imwrite(tmp_mp4, frames, fps=fps, codec="libx264", quality=8)
    if store_last:
        # как "последнюю картинку" сохраняем исходную (не первый кадр)
        try:
            _save_last_image(image)
        except Exception as e:
            print("save last image failed:", e)
    try:
        if os.path.exists(LAST_VIDEO_PATH):
            os.remove(LAST_VIDEO_PATH)
        os.replace(tmp_mp4, LAST_VIDEO_PATH)
        mp4_path = LAST_VIDEO_PATH
    except Exception:
        mp4_path = tmp_mp4  # fallback

    return {"video_b64": _encode_file_b64(mp4_path)}

runpod.serverless.start({"handler": handler})
