import os, subprocess, base64, glob, tempfile, urllib.request, time
import runpod

WAN_REPO = os.environ.get("WAN_REPO", "/workspace/Wan2.2")
CKPT_DIR = os.environ.get("WAN_CKPT_DIR", "/models/Wan2.2-TI2V-5B")
WARMUP   = os.environ.get("WARMUP", "0") == "1"
PIXEL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    "ASsJTYQAAAAASUVORK5CYII="
)

def _save_input_image(inp):
    """
    Поддержка:
    - image: "https://..." (скачиваем)
    - image: "data:image/...;base64,...."
    - image: "/path/inside/container.png"
    """
    img = inp.get("image")
    if not img:
        return None
    if isinstance(img, str) and img.startswith("data:"):
        b64 = img.split(",", 1)[1]
        p = os.path.join(tempfile.gettempdir(), "input.png")
        with open(p, "wb") as f:
            f.write(base64.b64decode(b64))
        return p
    if isinstance(img, str) and (img.startswith("http://") or img.startswith("https://")):
        p = os.path.join(tempfile.gettempdir(), "input.png")
        urllib.request.urlretrieve(img, p)
        return p
    return img  # считаем, что это путь внутри контейнера

def _run_generate(task, size, prompt, image_path, extra_args=None):
    """
    Запуск их скрипта через CLI. TI2V-5B покрывает T2V/I2V, но для I2V нужна картинка.
    size '1280*704' — «720p-friendly» для TI2V-5B.
    """
    cmd = [
        "python", f"{WAN_REPO}/generate.py",
        "--task", task,
        "--size", size,
        "--ckpt_dir", CKPT_DIR,
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu"
    ]
    if prompt not in (None, ""):
        cmd += ["--prompt", str(prompt)]
    if image_path:
        cmd += ["--image", image_path]
    if isinstance(extra_args, list):
        cmd += extra_args

    subprocess.check_call(cmd)

    results_dir = os.path.join(WAN_REPO, "results")
    latest = sorted(
        [os.path.join(results_dir, d) for d in os.listdir(results_dir)],
        key=os.path.getmtime
    )[-1]
    mp4s = sorted(glob.glob(os.path.join(latest, "*.mp4")))
    return mp4s[-1] if mp4s else None

def _warmup_once():
    """Одноразовый быстрый прогрев: tiny video на 0.5-1c из 1x1 PNG."""
    try:
        tmp = os.path.join(tempfile.gettempdir(), "warmup.png")
        with open(tmp, "wb") as f:
            f.write(base64.b64decode(PIXEL_PNG_B64))
        # короткая генерация: размер 256*144, меньше кадров
        _run_generate(
            task="ti2v-5B",
            size="256*144",
            prompt="warmup",
            image_path=tmp,
            extra_args=["--num_frames", "8"]
        )
    except Exception as e:
        print("Warmup failed:", e)
if WARMUP:
    _warmup_once()

def handler(job):
    inp = job.get("input", {})
    task = inp.get("task", "ti2v-5B")        
    size = inp.get("size", "1280*704")      
    prompt = inp.get("prompt", "")     
    image_path = _save_input_image(inp)
    extra = []

    if "num_frames" in inp:
        extra += ["--num_frames", str(int(inp["num_frames"]))]
    if "fps" in inp:
        extra += ["--fps", str(int(inp["fps"]))]

    if not image_path:
        return {"error": "image is required for I2V. Provide 'image' as URL, path, or data:base64."}

    out_mp4 = _run_generate(task, size, prompt, image_path, extra_args=extra)
    if not out_mp4:
        return {"error": "no video produced"}

    with open(out_mp4, "rb") as f:
        v64 = base64.b64encode(f.read()).decode()
    return {"video_b64": v64}

# стартуем воркер
runpod.serverless.start({"handler": handler})
