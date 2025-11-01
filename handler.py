import os, io, base64, tempfile, urllib.request
from typing import Optional, List
from PIL import Image
import imageio.v3 as iio
import numpy as np
import torch, runpod
from diffusers import I2VGenXLPipeline, DPMSolverMultistepScheduler

# ---------------------------
# Авто-детект пути к Network Volume
# ---------------------------
def detect_data_dir() -> str:
    # приоритет: явный ENV → стандартные точки монтирования
    env_dir = os.getenv("DATA_DIR")
    candidates = [env_dir, "/data", "/runpod-volume", "/workspace"]
    tried = []
    for base in candidates:
        if not base:
            continue
        model_dir = os.path.join(base, "models", "i2vgen-xl")
        tried.append(model_dir)
        if os.path.isdir(model_dir):
            # признак валидной модели — наличие model_index.json
            if os.path.isfile(os.path.join(model_dir, "model_index.json")):
                print(f"[init] Using model at: {model_dir}")
                return base
    # если не нашли — кинем понятную ошибку с подсказкой
    raise RuntimeError(
        "Model files not found. Looked at:\n  - " + "\n  - ".join(tried) +
        "\nPlace 'ali-vilab/i2vgen-xl' into your Network Volume under "
        "'<mount>/models/i2vgen-xl' and set DATA_DIR=<mount> if needed."
    )

# --- оффлайн режим ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

DATA_DIR  = detect_data_dir()
MODEL_DIR = os.path.join(DATA_DIR, "models", "i2vgen-xl")
CACHE_DIR = os.environ.get("CACHE_DIR", "/cache")
WARMUP    = os.environ.get("WARMUP", "0") == "1"
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", "0") == "1"
os.makedirs(CACHE_DIR, exist_ok=True)

LAST_IMAGE_PATH = os.path.join(CACHE_DIR, "last_image.png")
LAST_VIDEO_PATH = os.path.join(CACHE_DIR, "last_video.mp4")

# ---------------------------
# Инициализация пайплайна (оффлайн)
# ---------------------------
DTYPE = torch.float16
device = "cuda"

pipe = I2VGenXLPipeline.from_pretrained(
    MODEL_DIR, torch_dtype=DTYPE, variant="fp16", local_files_only=True
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
if CPU_OFFLOAD:
    pipe.enable_model_cpu_offload()
else:
    pipe.to(device)
pipe.enable_vae_tiling()

# ---------------------------
# Утилиты
# ---------------------------
def _pil_from_any(s: str) -> Image.Image:
    if s.startswith("data:"):
        b64 = s.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if s.startswith("http://") or s.startswith("https://"):
        p = os.path.join(tempfile.gettempdir(), "i2v.png")
        urllib.request.urlretrieve(s, p)
        return Image.open(p).convert("RGB")
    return Image.open(s).convert("RGB")

def _save_last_image(img: Image.Image): img.save(LAST_IMAGE_PATH)

def _load_last_image() -> Optional[Image.Image]:
    if os.path.exists(LAST_IMAGE_PATH):
        try: return Image.open(LAST_IMAGE_PATH).convert("RGB")
        except: return None
    return None

def _encode_file_b64(path: str) -> str:
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode()

def _make_even(x: int) -> int:
    return x if x % 2 == 0 else x - 1  # для yuv420p нужны чётные размеры

def _warmup_once():
    try:
        px = Image.new("RGB", (256, 256), (255, 255, 255))
        _save_last_image(px)
        _ = pipe(
            prompt="warmup",
            image=px,
            num_frames=8,
            guidance_scale=5.0,
            height=256,
            width=256,
            generator=torch.Generator(device=device).manual_seed(42)
        ).frames
        print("[warmup] done")
    except Exception as e:
        print("[warmup] failed:", e)

if WARMUP:
    _warmup_once()

# ---------------------------
# Handler
# ---------------------------
def handler(job):
    """
    input:
      image: URL / data:base64 / path (не обяз., если use_last_image=true)
      prompt: str
      num_frames: int (def 16)
      fps: int (def 8)
      width,height: int (def 512,512)
      guidance_scale: float (def 7.0)
      seed: int (опц.)
      use_last_image: bool (def false)
      store_last: bool (def true)
      action: "get_last_image" | "clear_cache"
    """
    inp = job.get("input", {}) or {}

    # сервисные экшены
    action = inp.get("action")
    if action == "get_last_image":
        img = _load_last_image()
        if not img: return {"error": "no last image stored"}
        bio = io.BytesIO(); img.save(bio, format="PNG")
        return {"image_b64": base64.b64encode(bio.getvalue()).decode()}

    if action == "clear_cache":
        cleared=[]
        for p in (LAST_IMAGE_PATH, LAST_VIDEO_PATH):
            if os.path.exists(p): os.remove(p); cleared.append(os.path.basename(p))
        return {"cleared": cleared}

    use_last = bool(inp.get("use_last_image", False))
    store_last = True if inp.get("store_last", True) else False

    if use_last:
        image = _load_last_image()
        if image is None and inp.get("image"):
            image = _pil_from_any(inp["image"])
        if image is None:
            return {"error":"no last image stored; provide 'image' or set use_last_image=false"}
    else:
        if not inp.get("image"):
            return {"error":"provide 'image' (URL / data:base64 / path) or set use_last_image=true"}
        image = _pil_from_any(inp["image"])

    prompt = inp.get("prompt", "")
    num_frames = int(inp.get("num_frames", 16))
    fps = max(1, int(inp.get("fps", 8)))  # защита от fps < 1
    width = _make_even(int(inp.get("width", 512)))
    height = _make_even(int(inp.get("height", 512)))
    guidance_scale = float(inp.get("guidance_scale", 7.0))
    seed = inp.get("seed")
    generator = torch.Generator(device=device).manual_seed(int(seed)) if seed else None

    # Генерация кадров
    frames_pil: List[Image.Image] = pipe(
        prompt=prompt,
        image=image,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator
    ).frames

    # Приводим каждый кадр к RGB и uint8 — иначе ffmpeg/VideoWriter может ругаться
    frames_np = [np.asarray(f.convert("RGB"), dtype=np.uint8) for f in frames_pil]

    tmp_mp4 = os.path.join(tempfile.gettempdir(), "out.mp4")
    # Используем yuv420p (широко совместим) и quality=8
    iio.imwrite(tmp_mp4, frames_np, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8)

    if store_last:
        try: _save_last_image(image)
        except Exception as e: print("save last image failed:", e)

    try:
        if os.path.exists(LAST_VIDEO_PATH): os.remove(LAST_VIDEO_PATH)
        os.replace(tmp_mp4, LAST_VIDEO_PATH); mp4_path = LAST_VIDEO_PATH
    except Exception:
        mp4_path = tmp_mp4

    return {"video_b64": _encode_file_b64(mp4_path)}

runpod.serverless.start({"handler": handler})
