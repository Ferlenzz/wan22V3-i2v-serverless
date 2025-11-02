import os, io, base64, tempfile, subprocess, urllib.request, shutil
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

def find_first(root: str, predicates: list) -> Optional[str]:
    for dirpath, _, files in os.walk(root):
        for fn in files:
            name = fn.lower()
            for pred in predicates:
                if pred(name):
                    return os.path.join(dirpath, fn)
    return None

def autodetect_vc2_script_and_cfg(vc2_root: str) -> tuple[str, str]:
    # Скрипт: .sh > image2video.py > i2v*.py
    script = find_first(vc2_root, [
        lambda n: n == "run_image2video.sh",
        lambda n: "image2video" in n and n.endswith(".py"),
        lambda n: "i2v" in n and n.endswith(".py"),
    ])
    # Конфиг: сначала 512, потом любой i2v
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

# Выставим HF кэш на NV (если не задано)
HF_HOME = os.environ.get("HF_HOME", os.path.join(DATA_DIR, "hf_cache"))
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HOME

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
# HF offline cache for open_clip
# ---------------------------
def ensure_openclip_cached():
    """
    Кладём CLIP-вес в структуру HF cache, чтобы open_clip/hf_hub_download нашёл локально (offline).
    Ждёт файл: <DATA_DIR>/models/clip/open_clip_pytorch_model.bin
    """
    src = os.path.join(DATA_DIR, "models", "clip", "open_clip_pytorch_model.bin")
    if not os.path.isfile(src):
        raise RuntimeError(
            f"Missing CLIP weight: {src}\n"
            "Place 'open_clip_pytorch_model.bin' into <DATA_DIR>/models/clip/ ."
        )

    model_id_dir = os.path.join(HF_HOME, "hub", "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K")
    refs_dir     = os.path.join(model_id_dir, "refs")
    snap_dir     = os.path.join(model_id_dir, "snapshots", "local")  # фиктивный снапшот 'local'
    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(snap_dir, exist_ok=True)

    # refs/main -> 'local'
    with open(os.path.join(refs_dir, "main"), "w") as f:
        f.write("local")

    dst = os.path.join(snap_dir, "open_clip_pytorch_model.bin")
    if not os.path.isfile(dst):
        shutil.copy2(src, dst)
    print(f"[hf-cache] CLIP cached at: {dst}")

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
# Вызов VideoCrafter2 (i2v) с авто-ретраями по флагам
# ---------------------------
def _build_args(alias: dict, input_png: str, prompt: str, out_dir: str,
                num_frames: int, fps: int, steps: int, guidance: float, seed: Optional[int]) -> list[str]:
    # alias: подмена имён флагов для разных ревизий (например frames->num_frames, save_dir->output_dir, image->input)
    return [
        "--config", VC2_CFG,
        "--ckpt",   MODEL_CKPT,
        alias.get("image", "--image"),  input_png,
        alias.get("prompt", "--prompt"), prompt,
        alias.get("save_dir", "--save_dir"), out_dir,
        alias.get("fps", "--fps"), str(fps),
        alias.get("frames", "--frames"), str(num_frames),
        alias.get("steps", "--num_inference_steps"), str(steps),
        alias.get("guidance", "--guidance_scale"), str(guidance),
    ] + (([alias.get("seed", "--seed"), str(seed)] if seed is not None else []))

def run_vc2_i2v(img_path: str, prompt: str, out_dir: str,
               num_frames: int, fps: int, steps: int, guidance: float, seed: Optional[int]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    input_png = os.path.join(out_dir, "input.png")
    Image.open(img_path).convert("RGB").save(input_png)

    # Возможные варианты названий флагов у разных ревизий скриптов
    aliases = [
        {},  # по умолчанию: --image/--save_dir/--frames/--num_inference_steps
        {"frames": "--num_frames"},  # иногда frames -> num_frames
        {"save_dir": "--output_dir"},  # иногда save_dir -> output_dir
        {"save_dir": "--output"},      # save_dir -> output
        {"image": "--input"},          # image -> input
        {"frames": "--nframes"},       # frames -> nframes
    ]

    # Определяем интерпретатор и корень репозитория для корректных относительных путей
    if VC2_SCRIPT.endswith(".sh"):
        launcher = ["bash", VC2_SCRIPT]
    else:
        launcher = ["python", VC2_SCRIPT]

    script_dir = os.path.dirname(VC2_SCRIPT)
    repo_root = script_dir if os.path.basename(script_dir) != "scripts" else os.path.dirname(script_dir)

    last_out = ""
    for alias in aliases:
        args = _build_args(alias, input_png, prompt, out_dir, num_frames, fps, steps, guidance, seed)
        cmd = launcher + args
        print("[vc2] script:", VC2_SCRIPT)
        print("[vc2] cfg   :", VC2_CFG)
        print("[vc2] cwd   :", repo_root)
        print("[vc2] cmd   :", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_out = proc.stdout
        if proc.returncode == 0:
            return
        # Если ругается на неизвестный аргумент — пробуем следующий alias
        if "unrecognized arguments" in last_out or "unrecognised arguments" in last_out:
            print("[vc2] retry with different flags...")
            continue
        # Иначе это «реальная» ошибка — выходим
        break

    print(last_out)
    raise RuntimeError("VideoCrafter2 inference failed")

# ---------------------------
# Handler
# ---------------------------
def handler(job):
    ensure_openclip_cached()  # важно: подготавливаем офлайн-кэш CLIP

    inp = job.get("input", {}) or {}
    image = inp.get("image")
    if not image:
        return {"error": "provide 'image' (URL / data:base64 / path)"}
    prompt = inp.get("prompt", "")

    num_frames = int(inp.get("num_frames", 40))   # ~5 сек при fps=8
    fps        = int(inp.get("fps", 8))
    width      = int(inp.get("width", 512))
    height     = int(inp.get("height", 512))
    steps      = int(inp.get("num_inference_steps", 42))
    guidance   = float(inp.get("guidance_scale", 7.0))
    seed_raw   = inp.get("seed")
    seed       = int(seed_raw) if (seed_raw is not None and str(seed_raw).strip() != "") else None

    work = tempfile.mkdtemp(prefix="vc2i2v_")
    in_png = os.path.join(work, "input.png")
    _pil_from_any(image).resize((width, height), Image.LANCZOS).save(in_png)

    out_dir = os.path.join(work, "out")
    run_vc2_i2v(in_png, prompt, out_dir, num_frames, fps, steps, guidance, seed)

    # найдём видео (mp4)
    mp4 = None
    # чаще всего VC сохраняет результат под result.mp4/output.mp4
    for name in ("result.mp4", "output.mp4", "video.mp4"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            mp4 = p; break

    if not mp4:
        # если VC сохранил кадры — соберём сами
        frames = []
        for fn in sorted(os.listdir(out_dir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                frames.append(iio.imread(os.path.join(out_dir, fn)))
        if not frames:
            return {"error": "No mp4 or frames produced by VC2"}
        mp4 = os.path.join(out_dir, "result.mp4")
        iio.imwrite(mp4, frames, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8)

    with open(mp4, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    return {"video_b64": video_b64}

runpod.serverless.start({"handler": handler})
