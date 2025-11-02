import os, io, sys, json, base64, tempfile, subprocess, urllib.request
from typing import Optional, List
from PIL import Image
import imageio.v3 as iio
import runpod

# ---------------------------
# Авто-детект DATA_DIR (NV)
# ---------------------------
def detect_data_dir() -> str:
    env_dir = os.getenv("DATA_DIR")
    candidates = [env_dir, "/runpod-volume", "/data", "/workspace"]
    tried = []
    for base in candidates:
        if not base:
            continue
        model_dir = os.path.join(base, "models", "videocrafter2-i2v")
        tried.append(model_dir)
        if os.path.isdir(model_dir):
            print(f"[init] Using DATA_DIR={base}")
            return base
    raise RuntimeError(
        "Model dir not found. Checked:\n  - " + "\n  - ".join(tried) +
        "\nPut weights into '<NV>/models/videocrafter2-i2v/model.ckpt' and set DATA_DIR to <NV>."
    )

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

DATA_DIR = detect_data_dir()
MODEL_CKPT = os.path.join(DATA_DIR, "models", "videocrafter2-i2v", "model.ckpt")
if not os.path.isfile(MODEL_CKPT):
    raise RuntimeError(f"Checkpoint not found: {MODEL_CKPT}")

VC2_SCRIPT = os.environ.get("VC2_SCRIPT", "/vc2/run_image2video.sh")  # .sh или .py
VC2_CFG    = os.environ.get("VC2_CFG", "/vc2/configs/inference_i2v_512_v1.0.yaml")

CACHE_DIR  = os.environ.get("CACHE_DIR", "/cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------
# Утилиты
# ---------------------------
def _pil_from_any(s: str) -> Image.Image:
    if s.startswith("data:"):
        b64 = s.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if s.startswith("http://") or s.startswith("https://"):
        p = os.path.join(tempfile.gettempdir(), "i2v_input.png")
        urllib.request.urlretrieve(s, p)
        return Image.open(p).convert("RGB")
    return Image.open(s).convert("RGB")

def _encode_file_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------------------------
# Вызов VideoCrafter2 (i2v)
# ---------------------------
def run_vc2_i2v(img_path: str, prompt: str, out_dir: str,
               num_frames: int, fps: int, steps: int, guidance: float, seed: Optional[int]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Подготовим временный PNG, чтобы быть независимыми от формата
    input_png = os.path.join(out_dir, "input.png")
    Image.open(img_path).convert("RGB").save(input_png)

    # Собираем аргументы: в AILab-CVC/VideoCrafter есть run_image2video.sh,
    # который принимает --config/--ckpt/--image/--prompt/--save_dir/--fps/--frames/--num_inference_steps/--guidance_scale
    args_common = [
        "--config", VC2_CFG,
        "--ckpt",   MODEL_CKPT,
        "--image",  input_png,
        "--prompt", prompt,
        "--save_dir", out_dir,
        "--fps", str(fps),
        "--frames", str(num_frames),
        "--num_inference_steps", str(steps),
        "--guidance_scale", str(guidance),
    ]
    if seed is not None:
        args_common += ["--seed", str(seed)]

    # Выбор интерпретатора по расширению
    if VC2_SCRIPT.endswith(".sh"):
        cmd = ["bash", VC2_SCRIPT] + args_common
    else:
        # считаем, что это .py
        cmd = ["python", VC2_SCRIPT] + args_common

    print("[vc2] cmd:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError("VideoCrafter2 inference failed")

def handler(job):
    inp = job.get("input", {}) or {}

    image = inp.get("image")
    if not image:
        return {"error": "provide 'image' (URL / data:base64 / path)"}
    prompt = inp.get("prompt", "")

    num_frames = int(inp.get("num_frames", 40))   # ~5с при fps=8
    fps        = int(inp.get("fps", 8))
    width      = int(inp.get("width", 512))
    height     = int(inp.get("height", 512))
    steps      = int(inp.get("num_inference_steps", 42))
    guidance   = float(inp.get("guidance_scale", 7.0))
    seed       = inp.get("seed")
    seed       = int(seed) if seed is not None else None

    # Сохраняем вход и выход
    work = tempfile.mkdtemp(prefix="vc2i2v_")
    in_png = os.path.join(work, "input.png")
    _pil_from_any(image).resize((width, height)).save(in_png)

    out_dir = os.path.join(work, "out")
    run_vc2_i2v(in_png, prompt, out_dir, num_frames, fps, steps, guidance, seed)

    # найдём видео (mp4)
    mp4 = None
    for name in ("result.mp4", "output.mp4", "video.mp4"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            mp4 = p; break
    if not mp4:
        # если VC2 сохранил как images → соберём сами
        frames = []
        for fn in sorted(os.listdir(out_dir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                frames.append(Image.open(os.path.join(out_dir, fn)).convert("RGB"))
        if not frames:
            raise RuntimeError("No mp4 or frames produced")
        mp4 = os.path.join(out_dir, "result.mp4")
        iio.imwrite(mp4, frames, fps=fps, codec="libx264", quality=8)

    return {"video_b64": _encode_file_b64(mp4)}

runpod.serverless.start({"handler": handler})
