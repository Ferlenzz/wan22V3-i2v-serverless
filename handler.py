import os, io, sys, json, base64, tempfile, subprocess, shutil, glob
from typing import Optional, List
from PIL import Image
import imageio.v3 as iio
import runpod

# ---------------------------
# Полезные ENV/пути
# ---------------------------
DATA_DIR = os.environ.get("DATA_DIR") or "/runpod-volume"
HF_HOME  = os.environ.get("HF_HOME")  or os.path.join(DATA_DIR, "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HOME

CACHE_DIR = os.environ.get("CACHE_DIR", "/cache")
os.makedirs(CACHE_DIR, exist_ok=True)

VC1_SCRIPT = os.environ.get("VC1_SCRIPT", "/vc1/scripts/run_image2video.sh")
VC1_CFG    = os.environ.get("VC1_CFG", "/vc1/configs/inference_i2v_512_vc1_offline.yaml")

# где лежит офлайн CLIP из transformers
CLIP_DIR_DEFAULT = os.path.join(DATA_DIR, "models", "transformers", "openai-clip-vit-large-patch14")

LAST_IMAGE_PATH = os.path.join(CACHE_DIR, "last_image.png")
LAST_VIDEO_PATH = os.path.join(CACHE_DIR, "last_video.mp4")


# ---------------------------
# Вспомогательные функции
# ---------------------------
def _log(*a): print("[i2v]", *a, flush=True)

def _ensure_vc1_paths() -> (str, str, str):
    """Проверяем, что есть vc1, скрипт и конфиг. Возвращаем (repo_root, script, cfg)."""
    repo_root = os.path.dirname(os.path.dirname(VC1_SCRIPT)) if VC1_SCRIPT else "/vc1"
    script = VC1_SCRIPT
    cfg = VC1_CFG

    if not os.path.isfile(script):
        # пытаемся найти
        maybe = glob.glob("/vc1/**/run_image2video.sh", recursive=True) + glob.glob("/vc1/*.sh")
        script = next((p for p in maybe if "run_image2video" in os.path.basename(p)), None)
        if not script:
            raise RuntimeError("Не найден run_image2video.sh (ожидали /vc1/scripts/run_image2video.sh)")

    if not os.path.isfile(cfg):
        # подхватим любой inference_i2v_512*.yaml
        cand = glob.glob("/vc1/**/inference_i2v_512*.yaml", recursive=True)
        cfg = next(iter(cand), None)
        if not cfg:
            raise RuntimeError("Не найден конфиг inference_i2v_512*.yaml")

    return repo_root, script, cfg

def _ensure_checkpoint_link(repo_root: str):
    """
    Создаём 'жёсткий' путь чекпойнта внутри дерева VC1:
    /vc1/checkpoints/i2v_512_v1/model.ckpt -> /DATA_DIR/models/videocrafter2-i2v/model.ckpt
    """
    src = os.path.join(DATA_DIR, "models", "videocrafter2-i2v", "model.ckpt")
    if not os.path.isfile(src):
        raise RuntimeError(f"Чекпойнт не найден: {src}")

    ckpt_dir = os.path.join(repo_root, "checkpoints", "i2v_512_v1")
    os.makedirs(ckpt_dir, exist_ok=True)
    dst = os.path.join(ckpt_dir, "model.ckpt")
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
    except:
        pass
    os.symlink(src, dst)
    return dst

def _prepare_offline_clip_transformers():
    """
    Готовим офлайн snapshot для transformers CLIP (openai/clip-vit-large-patch14).
    Берём из DATA_DIR/models/transformers/openai-clip-vit-large-patch14/
    и размещаем в HF_HOME так, чтобы from_pretrained(local_files_only=True) всё находил.
    """
    src = os.environ.get("TRANSFORMERS_CLIP_DIR", "") or CLIP_DIR_DEFAULT
    if not os.path.isdir(src):
        _log(f"Внимание: папка CLIP transformers не найдена: {src}")
        return  # handler всё равно попробует работать; VC1 может не трогать from_pretrained в python — но лучше положить папку.

    # целевой каталог
    tgt_repo = os.path.join(HF_HOME, "hub", "models--openai--clip-vit-large-patch14")
    tgt_snap = os.path.join(tgt_repo, "snapshots", "offline")
    os.makedirs(tgt_snap, exist_ok=True)

    # минимальный набор файлов (если есть — копируем всё)
    for name in os.listdir(src):
        sp = os.path.join(src, name)
        dp = os.path.join(tgt_snap, name)
        if os.path.isdir(sp): 
            if not os.path.exists(dp): shutil.copytree(sp, dp, dirs_exist_ok=True)
        else:
            shutil.copy2(sp, dp)

    # refs/main -> offline
    refs_dir = os.path.join(tgt_repo, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    with open(os.path.join(refs_dir, "main"), "w") as f:
        f.write("offline")

    _log("Офлайн CLIP transformers оформлен:", tgt_repo)


def _pil_from_any(s: str) -> Image.Image:
    if s.startswith("data:"):
        b64 = s.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if s.startswith("http://") or s.startswith("https://"):
        p = os.path.join(tempfile.gettempdir(), "i2v_in.png")
        import urllib.request
        urllib.request.urlretrieve(s, p)
        return Image.open(p).convert("RGB")
    return Image.open(s).convert("RGB")

def _save_last_image(img: Image.Image):
    img.save(LAST_IMAGE_PATH)

def _load_last_image() -> Optional[Image.Image]:
    if os.path.exists(LAST_IMAGE_PATH):
        try:
            return Image.open(LAST_IMAGE_PATH).convert("RGB")
        except:
            return None
    return None

def _encode_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _ensure_mp4_from_frames(save_dir: str, fps: int) -> Optional[str]:
    # ищем mp4; если нет — собираем из кадров
    mp4s = sorted(glob.glob(os.path.join(save_dir, "*.mp4")))
    if mp4s: 
        return mp4s[0]

    frames = sorted(glob.glob(os.path.join(save_dir, "*.png")))
    if not frames:
        return None
    out_mp4 = os.path.join(save_dir, "out.mp4")
    imgs = [Image.open(p).convert("RGB") for p in frames]
    iio.imwrite(out_mp4, imgs, fps=fps, codec="libx264", quality=8)
    return out_mp4


# ---------------------------
# Основной запуск VC1 (shell)
# ---------------------------
def run_vc1(image: Image.Image, prompt: str, num_frames: int, fps: int,
            height: int, width: int, guidance_scale: float, steps: int,
            seed: Optional[int]) -> str:
    repo_root, script, cfg = _ensure_vc1_paths()
    _log("VC1 root:", repo_root)
    _log("script:", script)
    _log("cfg:", cfg)

    # чекпойнт
    ckpt = _ensure_checkpoint_link(repo_root)

    # офлайн CLIP transformers
    _prepare_offline_clip_transformers()

    # входное изображение
    os.makedirs(CACHE_DIR, exist_ok=True)
    in_png = os.path.join(CACHE_DIR, "vc1_input.png")
    image.save(in_png)

    # выходной каталог
    out_dir = tempfile.mkdtemp(prefix="vc1_out_", dir=CACHE_DIR)

    # строим команду (скрипт VC1 понимает --config, --ckpt, --image, --prompt, --save_dir, --n_frames, --num_inference_steps и пр.)
    # важное: некоторые ревизии ждут --frames вместо --n_frames — оставим оба
    cmd = [
        "bash", script,
        "--config", cfg,
        "--ckpt", ckpt,
        "--image", in_png,
        "--prompt", prompt,
        "--save_dir", out_dir,
        "--n_frames", str(num_frames),
        "--frames",   str(num_frames),    # на всякий
        "--num_inference_steps", str(steps),
        "--height", str(height),
        "--width",  str(width),
        "--fps",    str(fps),
        "--guidance_scale", str(guidance_scale),
    ]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]

    _log("run:", " ".join(cmd))
    env = {**os.environ}  # офлайн HF уже прописан
    proc = subprocess.run(cmd, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    _log("vc1 stdout:\n" + proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError("VideoCrafter inference failed")

    # результат
    mp4 = _ensure_mp4_from_frames(out_dir, fps=fps)
    if mp4 is None:
        raise RuntimeError("Не найдено видео/кадры после инференса")

    # перенос в стабильный путь
    if os.path.exists(LAST_VIDEO_PATH):
        try: os.remove(LAST_VIDEO_PATH)
        except: pass
    shutil.copy2(mp4, LAST_VIDEO_PATH)
    return LAST_VIDEO_PATH


# ---------------------------
# Warmup (опционально)
# ---------------------------
def _warmup():
    try:
        img = Image.new("RGB", (256, 256), (240, 240, 240))
        _save_last_image(img)
        _ = run_vc1(
            image=img, prompt="warmup",
            num_frames=4, fps=6, height=256, width=256,
            guidance_scale=5.0, steps=8, seed=42
        )
        _log("warmup done")
    except Exception as e:
        _log("warmup failed:", e)


# ---------------------------
# RunPod handler
# ---------------------------
def handler(job):
    """
    input:
      image: data:base64 / url / path (если use_last_image=false)
      prompt: str
      num_frames: int (def 24)
      fps: int (def 8)
      width,height: int (def 512,512)
      guidance_scale: float (def 7.0)
      num_inference_steps: int (def 42)
      seed: int (опц.)
      use_last_image: bool (def false)
      store_last: bool (def true)
      action: "get_last_image" | "clear_cache"
    """
    args = job.get("input", {}) or {}

    # сервисные экшены
    action = args.get("action")
    if action == "get_last_image":
        img = _load_last_image()
        if not img: return {"error": "no last image stored"}
        bio = io.BytesIO(); img.save(bio, format="PNG")
        return {"image_b64": base64.b64encode(bio.getvalue()).decode()}

    if action == "clear_cache":
        cleared=[]
        for p in (LAST_IMAGE_PATH, LAST_VIDEO_PATH):
            if os.path.exists(p):
                try: os.remove(p); cleared.append(os.path.basename(p))
                except: pass
        return {"cleared": cleared}

    use_last = bool(args.get("use_last_image", False))
    store_last = True if args.get("store_last", True) else False

    if use_last:
        image = _load_last_image()
        if image is None and args.get("image"):
            image = _pil_from_any(args["image"])
        if image is None:
            return {"error":"no last image stored; provide 'image' or set use_last_image=false"}
    else:
        if not args.get("image"):
            return {"error":"provide 'image' (URL / data:base64 / path) or set use_last_image=true"}
        image = _pil_from_any(args["image"])

    prompt = args.get("prompt", "")
    num_frames = int(args.get("num_frames", 24))
    fps = int(args.get("fps", 8))
    width = int(args.get("width", 512))
    height = int(args.get("height", 512))
    guidance_scale = float(args.get("guidance_scale", 7.0))
    steps = int(args.get("num_inference_steps", 42))
    seed = args.get("seed")
    if seed is not None:
        try: seed = int(seed)
        except: seed = None

    # запуск
    out_mp4 = run_vc1(
        image=image, prompt=prompt, num_frames=num_frames, fps=fps,
        height=height, width=width, guidance_scale=guidance_scale,
        steps=steps, seed=seed
    )

    if store_last:
        try: _save_last_image(image)
        except: pass

    return {"video_b64": _encode_b64(out_mp4)}

# Если хочешь включить прогрев при старте воркера — раскомментируй:
# _warmup()

runpod.serverless.start({"handler": handler})
