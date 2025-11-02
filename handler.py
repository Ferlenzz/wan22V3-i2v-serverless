# handler.py
# Полный, самодостаточный обработчик для Runpod + VideoCrafter (VC1, image2video).
# Запускает скрипт VC1 из корня репозитория (/vc1) с корректным cwd,
# принимает картинку (data:base64 / URL / локальный путь) + prompt и возвращает mp4 в base64.

import os
import io
import re
import json
import base64
import shutil
import tempfile
import subprocess
from typing import Optional
from urllib.request import urlretrieve

try:
    import runpod
except Exception:  # локальный запуск без runpod
    class _RP:
        class serverless:
            @staticmethod
            def start(_):  # no-op
                pass
    runpod = _RP()


# -----------------------------
# Конфигурация путей / ENV
# -----------------------------
def detect_data_dir() -> str:
    for p in [
        os.environ.get("DATA_DIR"),
        "/runpod-volume",
        "/data",
        "/workspace",
    ]:
        if p and os.path.isdir(p):
            return p
    return "/runpod-volume"  # по умолчанию (если не смонтирован — просто временно)


DATA_DIR = detect_data_dir()

VC1_ROOT = os.environ.get("VC1_ROOT", "/vc1").rstrip("/")
VC1_SCRIPT = os.environ.get("VC1_SCRIPT", f"{VC1_ROOT}/scripts/run_image2video.sh")
VC1_CFG = os.environ.get("VC1_CFG", f"{VC1_ROOT}/configs/inference_i2v_512_vc1_offline.yaml")
# чекпоинт VC1 — можно переопределить через ENV VC1_CKPT
VC1_CKPT = os.environ.get("VC1_CKPT", f"{DATA_DIR}/checkpoints/i2v_512_v1/model.ckpt")

CACHE_DIR = os.environ.get("CACHE_DIR", "/cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# -----------------------------
# Утилиты
# -----------------------------
def _is_data_url(s: str) -> bool:
    return s.startswith("data:") and ";base64," in s

def _save_input_image(value: str) -> str:
    """
    value: data URL, http(s) URL, абсолютный/относительный путь.
    Возвращает путь до PNG/JPG на локальном диске.
    """
    if not value:
        raise RuntimeError("image is required")

    tmpdir = tempfile.mkdtemp(prefix="i2v_in_")
    out = os.path.join(tmpdir, "input.png")

    if _is_data_url(value):
        b64 = value.split(",", 1)[1]
        with open(out, "wb") as f:
            f.write(base64.b64decode(b64))
        return out

    if value.startswith("http://") or value.startswith("https://"):
        urlretrieve(value, out)
        return out

    # путь в ФС
    if os.path.isfile(value):
        # скопируем, чтобы не ломать оригинал
        shutil.copy(value, out)
        return out

    raise RuntimeError(f"Unsupported image value: {value}")

def _ensure_file(path: str, name: str = "file"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} not found: {path}")

def _find_latest_mp4(root: str) -> Optional[str]:
    latest = None
    latest_mtime = -1.0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".mp4"):
                full = os.path.join(dirpath, fn)
                try:
                    mtime = os.path.getmtime(full)
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest = full
                except Exception:
                    pass
    return latest

def _b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# -----------------------------
# VC1 launcher (НОРМАЛЬНЫЙ СПОСОБ)
# -----------------------------
def run_vc1(
    image_path: str,
    prompt: str,
    frames: int = 24,
    steps: int = 42,
    height: int = 512,
    width: int = 512,
    fps: int = 8,
    guidance_scale: float = 7.0,
    work_id: Optional[str] = None,
) -> str:
    """
    Запускает VC1 из корня /vc1 (cwd=/vc1), чтобы относительные пути внутри скрипта работали.
    Возвращает путь к mp4.
    """
    # sanity checks
    _ensure_file(VC1_SCRIPT, "VC1 script")
    _ensure_file(VC1_CFG, "VC1 config")
    _ensure_file(VC1_CKPT, "VC1 checkpoint")
    _ensure_file(image_path, "input image")

    save_dir = os.path.join(CACHE_DIR, f"vc1_out_{work_id or 'job'}")
    os.makedirs(save_dir, exist_ok=True)

    # ВАЖНО: скрипт запускаем из /vc1 и используем ОТНОСИТЕЛЬНЫЙ путь "scripts/run_image2video.sh"
    # Аргументы — абсолютные, чтобы исключить двусмысленности.
    cmd = [
        "bash",
        "scripts/run_image2video.sh",
        "--config", os.path.relpath(VC1_CFG, VC1_ROOT),
        "--ckpt", VC1_CKPT,
        "--image", image_path,
        "--prompt", prompt,
        "--save_dir", save_dir,
        "--frames", str(frames),
        "--num_inference_steps", str(steps),
        "--height", str(height),
        "--width", str(width),
        "--fps", str(fps),
        "--guidance_scale", str(guidance_scale),
    ]

    # Запускаем и логируем stdout/stderr
    proc = subprocess.run(
        cmd,
        cwd=VC1_ROOT,                # КРИТИЧЕСКИ ВАЖНО: корень VC1
        capture_output=True,
        text=True,
        env={**os.environ},          # наследуем окружение
    )

    print("[VC1 stdout]\n", proc.stdout)
    print("[VC1 stderr]\n", proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError("VideoCrafter inference failed")

    # Ищем последнее mp4 в save_dir
    mp4 = _find_latest_mp4(save_dir)
    if not mp4:
        raise RuntimeError("Video not found after VC1 run")
    return mp4


# -----------------------------
# Runpod handler
# -----------------------------
def handler(event):
    """
    input:
      image: base64 (data URL) / http(s) / путь
      prompt: str
      frames: int (24)
      steps: int (42)
      height: int (512)
      width: int (512)
      fps: int (8)
      guidance_scale: float (7.0)
    """
    try:
        inp = event.get("input", {}) if isinstance(event, dict) else {}
        image = inp.get("image")
        prompt = inp.get("prompt", "")
        if not prompt:
            prompt = ""
        frames = int(inp.get("frames", inp.get("num_frames", 24)))
        steps = int(inp.get("steps", inp.get("num_inference_steps", 42)))
        height = int(inp.get("height", 512))
        width = int(inp.get("width", 512))
        fps = int(inp.get("fps", 8))
        guidance = float(inp.get("guidance_scale", 7.0))

        # подготовим входное изображение
        img_path = _save_input_image(image)

        # запустим VC1
        job_id = (event.get("id") if isinstance(event, dict) else None) or "job"
        mp4_path = run_vc1(
            image_path=img_path,
            prompt=prompt,
            frames=frames,
            steps=steps,
            height=height,
            width=width,
            fps=fps,
            guidance_scale=guidance,
            work_id=str(job_id),
        )

        return {
            "ok": True,
            "video_b64": _b64_file(mp4_path),
            "meta": {
                "data_dir": DATA_DIR,
                "vc1_root": VC1_ROOT,
                "cfg": VC1_CFG,
                "ckpt": VC1_CKPT,
                "frames": frames,
                "steps": steps,
                "size": [width, height],
                "fps": fps,
                "guidance_scale": guidance,
            },
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


# Запуск runpod
runpod.serverless.start({"handler": handler})
