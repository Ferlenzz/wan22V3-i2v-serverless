import os, io, base64, tempfile, subprocess, shlex, urllib.request
from PIL import Image
import imageio.v3 as iio
import runpod

def detect_data_dir() -> str:
    env = os.getenv("DATA_DIR")
    for base in [env, "/data", "/runpod-volume", "/workspace"]:
        if not base: continue
        ckpt = os.path.join(base, "models", "videocrafter2-i2v", "model.ckpt")
        if os.path.isfile(ckpt):
            print(f"[init] Using DATA_DIR={base}")
            return base
    raise RuntimeError("VC2 I2V .ckpt not found. Expected <DATA>/models/videocrafter2-i2v/model.ckpt")

def _pil_from_any(s: str):
    if s.startswith("data:"):
        b64 = s.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if s.startswith("http://") or s.startswith("https://"):
        p = os.path.join(tempfile.gettempdir(), "ref.png")
        urllib.request.urlretrieve(s, p)
        return Image.open(p).convert("RGB")
    return Image.open(s).convert("RGB")

def _encode_file_b64(path: str) -> str:
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)

DATA_DIR = detect_data_dir()
CACHE_DIR = os.environ.get("CACHE_DIR", "/cache"); os.makedirs(CACHE_DIR, exist_ok=True)
LAST_MP4 = os.path.join(CACHE_DIR, "vc2_i2v.mp4")

VC2_DIR    = "/vc2"
VC2_SCRIPT = os.environ.get("VC2_SCRIPT", os.path.join(VC2_DIR, "scripts", "inference_i2v.py"))
VC2_CFG    = os.environ.get("VC2_CFG",    os.path.join(VC2_DIR, "configs", "inference", "image2video_512.yaml"))
CKPT_PATH  = os.path.join(DATA_DIR, "models", "videocrafter2-i2v", "model.ckpt")

def run_vc2_i2v(image_path: str, prompt: str, out_dir: str, num_frames: int, fps: int, steps: int, guidance: float, seed: int | None):
    cmd = [
        "python", VC2_SCRIPT,
        "--config", VC2_CFG,
        "--ckpt", CKPT_PATH,
        "--image", image_path,               # если у твоей версии --input, поменяешь здесь
        "--prompt", prompt,
        "--save_dir", out_dir,               # иногда --output_dir
        "--fps", str(fps),
        "--num_frames", str(num_frames),
        "--num_inference_steps", str(steps),
        "--guidance_scale", str(guidance),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    print("[vc2] cmd:", " ".join(shlex.quote(x) for x in cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError("VideoCrafter2 inference failed")

def handler(job):
    inp = job.get("input", {}) or {}
    if not inp.get("image"):  return {"error":"provide 'image'"}
    if not inp.get("prompt"): return {"error":"provide 'prompt'"}

    prompt  = inp["prompt"]
    num_frames = int(inp.get("num_frames", 40))
    fps        = max(1, int(inp.get("fps", 8)))
    width      = int(inp.get("width", 512))
    height     = int(inp.get("height", 512))
    steps      = int(inp.get("num_inference_steps", 42))
    guidance   = float(inp.get("guidance_scale", 7.0))
    seed       = inp.get("seed")
    quality    = int(inp.get("quality", 2))

    img = _pil_from_any(inp["image"]).resize((width, height), Image.LANCZOS)
    tmp_dir  = tempfile.mkdtemp(prefix="vc2i2v_")
    img_path = os.path.join(tmp_dir, "input.png")
    out_dir  = os.path.join(tmp_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    img.save(img_path)

    run_vc2_i2v(img_path, prompt, out_dir, num_frames, fps, steps, guidance, int(seed) if seed else None)

    mp4_path = None
    for name in ["result.mp4", "output.mp4", "i2v.mp4"]:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            mp4_path = p; break

    if mp4_path is None:
        frames_dir = os.path.join(out_dir, "frames")
        if os.path.isdir(frames_dir):
            frames = []
            for fname in sorted(os.listdir(frames_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    frames.append(iio.imread(os.path.join(frames_dir, fname)))
            if not frames:
                return {"error":"no output frames found"}
            tmp_mp4 = os.path.join(tmp_dir, "collected.mp4")
            iio.imwrite(tmp_mp4, frames, fps=fps, codec="libx264", pixelformat="yuv420p", quality=quality)
            mp4_path = tmp_mp4

    if mp4_path is None:
        return {"error":"VC2 finished but no output mp4/frames. Adjust VC2_SCRIPT/VC2_CFG or args."}

    try:
        if os.path.exists(LAST_MP4): os.remove(LAST_MP4)
        os.replace(mp4_path, LAST_MP4); out = LAST_MP4
    except Exception:
        out = mp4_path

    return {"video_b64": _encode_file_b64(out)}

runpod.serverless.start({"handler": handler})
