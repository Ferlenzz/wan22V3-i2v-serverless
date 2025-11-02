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

def file_exists(p: Optional[str]) -> bool:
    return bool(p) and os.path.isfile(p)

def find_first(root: str, patterns: list[str]) -> Optional[str]:
    for dirpath, _, files in os.walk(root):
        for fn in files:
            name = fn.lower()
            for pat in patterns:
                if pat(name):
                    return os.path.join(dirpath, fn)
    return None

def autodetect_vc2_script_and_cfg(vc2_root: str) -> tuple[str, str]:
    # 1) Скрипт: пробуем .sh, потом .py
    script = find_first(vc2_root, [
        lambda n: n == "run_image2video.sh",
        lambda n: "image2video" in n and n.endswith(".py"),
        lambda n: "i2v" in n and n.endswith(".py"),
    ])
    # 2) Конфиг: сперва 512, потом любой i2v
    cfg = find_first(vc2_root, [
        lambda n: n.startswith("inference_i2v") and "512" in n and n.endswith(".yaml"),
        lambda n: n.startswith("inference_i2v") and n.endswith(".yaml"),
    ])
    if not script or not cfg:
        raise RuntimeError(
            "Unable to auto-detect VC2 script/cfg.\n"
            f"script={script}\n cfg={cfg}\n"
            "Check that your /vc2 tree contains run_image2video.sh or *image2video*.py, "
            "and configs/inference_i2v*.yaml"
        )
    return script, cfg

# --- оффлайн режим ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

DATA_DIR = detect_data_dir()
MODEL_CKPT = os.path.join(DATA_DIR, "models", "videocrafter2-i2v", "model.ckpt")
if not os.path.isfile(MODEL_CKPT):
    raise RuntimeError(f"Checkpoint not found: {MODEL_CKPT}")

# Берём из ENV, но если файла нет — автопоиск
VC2_SCRIPT_ENV = os.environ.get("VC2_SCRIPT", "")
VC2_CFG_ENV    = os.environ.get("VC2_CFG", "")
if file_exists(VC2_SCRIPT_ENV) and file_exists(VC2_CFG_ENV):
    VC2_SCRIPT, VC2_CFG = VC2_SCRIPT_ENV, VC2_CFG_ENV
else:
    VC2_SCRIPT, VC2_CFG = autodetect_vc2_script_and_cfg("/vc2")

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

    input_png = os.path.join(out_dir, "input.png")
    Image.open(img_path).convert("RGB").save(input_png)

    # Общие аргументы (названия флагов такие у run_image2video.sh и большинства image2video.py в VC)
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

    # Выбор интерпретатора
    if VC2_SCRIPT.endswith(".sh"):
        cmd = ["bash", VC2_SCRIPT] + args_common
    else:
        cmd = ["python", VC2_SCRIPT] + args_common

    print("[vc2] script:", VC2_SCRIPT)
    print("[vc2] cfg   :", VC2_CFG)
    print("[vc2] cmd   :", " ".join(cmd))
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

    num_frames = int(inp.get("num_frames", 40))
    fps        = int(inp.get("fps", 8))
    width      = int(inp.get("width", 512))
    height     = int(inp.get("height", 512))
    steps      = int(inp.get("num_inference_steps", 42))
    guidance   = float(inp.get("guidance_scale", 7.0))
    seed       = inp.get("seed")
    seed       = int(seed) if seed is not None else None

    work = tempfile.mkdtemp(prefix="vc2i2v_")
    in_png = os.path.join(work, "input.png")
    _pil_from_any(image).resize((width, height)).save(in_png)

    out_dir = os.path.join(work, "out")
    run_vc2_i2v(in_png, prompt, out_dir, num_frames, fps, steps, guidance, seed)

    # попробуем найти готовый mp4
    mp4 = None
    for name in ("result.mp4", "output.mp4", "video.mp4"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            mp4 = p; break
    if not mp4:
        # соберём видео из кадров, если их нагенерили
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
