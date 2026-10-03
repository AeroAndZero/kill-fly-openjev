import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from PIL import Image, UnidentifiedImageError

from decider.engine import patch_conv
from decider.vision import VisionDecisionModel

# CONFIG
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 72
USE_COMPILE = True            # flip off to compare; measure, don't assume


# replaces: from decider.infer import Example
@dataclass
class ImageQuestion:
    context: str
    qs: list
    task: str = "infer"
    image: bytes | None = None


# replaces: from decider.infer import Q
@dataclass
class Q:
    text: str
    options: list
    gold: int = 0


class ImageDecider:
    CONTEXT = "This is an image of a red ball in a room."
    QUESTIONS = {
        "position": ("Where is the red ball in the image?",
                     ["top left", "top center", "top right",
                      "center left", "center", "center right",
                      "bottom left", "bottom center", "bottom right"]),
        "visible":  ("Is red ball visible?", ["yes", "no"]),
        "distance": ("distance of red ball",
                     ["very close", "close", "medium range", "far", "very far"]),
    }

    TEMPERATURE = 1.0

    def __init__(self):
        # Swap Qwen3.5's depthwise causal conv for the package's fused version (what decider's own engine does).
        # Must run before the first forward.
        patch_conv()

        self.MODEL = VisionDecisionModel("Mapika/decider-2b-vision", dtype=torch.bfloat16,
                                         grad_ckpt=False).cuda().eval()
        dev = self.MODEL.letters.device

        # One example per question, so each is scored in its own row and can't influence the others
        self.EXAMPLES = [(key, opts, ImageQuestion(self.CONTEXT, [Q(text, opts, 0)]))
                         for key, (text, opts) in self.QUESTIONS.items()]
        self.N_OPTS = torch.tensor([len(opts) for _, opts, _ in self.EXAMPLES])

        # The text never changes and every image is resized to the same size, so the token ids, masks, image grid
        # and answer-slot positions are identical for every request. Build them once; per request only the pixels change.
        dummy = Image.new("RGB", (IMAGE_SIZE_X, IMAGE_SIZE_Y))
        tmpl = self.MODEL.prepare([(dummy, ex) for _, _, ex in self.EXAMPLES])
        self.TEMPLATE = {k: v.to(dev) for k, v in tmpl.items() if k != "pixel_values"}

        if USE_COMPILE:
            # VisionDecisionModel has no forward(): slot_logits() calls self.lm.model(...) directly, so
            # compiling self.MODEL itself never runs. Compile the language model, which does almost all the work.
            # Shapes are now fixed, so dynamic=False avoids recompiles.
            self.MODEL.lm.model.language_model.compile(dynamic=False)

        # Warm up so compilation / kernel autotuning happens at startup, not on the first real request.
        for _ in range(3):
            self.infer(dummy)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def infer(self, img: Image.Image) -> dict:
        pix = self.MODEL.proc.image_processor(images=[img] * len(self.EXAMPLES), return_tensors="pt")
        if not torch.equal(pix["image_grid_thw"].to(self.TEMPLATE["image_grid_thw"].device),
                           self.TEMPLATE["image_grid_thw"]):
            raise ValueError("image size changed; the cached prompt no longer matches")  # guards the cache

        inp = dict(self.TEMPLATE)
        inp["pixel_values"] = pix["pixel_values"]
        logits = self.MODEL.slot_logits(inp)                   # [3, MAX_OPTIONS], unused options already -inf

        # One softmax and one device->host copy for all three questions (was three syncs).
        probs = torch.softmax(logits / self.TEMPERATURE, dim=-1).cpu()

        answers = {}
        for row, (key, opts, _) in zip(probs, self.EXAMPLES):
            answers[key] = sorted(zip(row[: len(opts)].tolist(), opts), reverse=True)  # [(prob, option), ...]

        pos, vis, dist = answers["position"], answers["visible"], answers["distance"]
        return {
            "positions": [{"label": label, "probs": p} for p, label in pos],
            "visibles": [{"label": label, "probs": p} for p, label in vis],      # "yes" / "no"
            "distances": [{"label": label, "probs": p} for p, label in dist],
        }


# Main server
app = FastAPI()
imageDecider = ImageDecider()

# All GPU work goes through ONE dedicated thread:
#  - the async event loop never blocks on inference, so other sockets keep receiving/decoding meanwhile
#  - requests from multiple connections are serialized on the single model (no concurrent forwards)
GPU_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


class BadImage(Exception):
    pass


def to_pil(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        # JPEG only: decode at 1/2, 1/4 or 1/8 scale (never below the target). Big phone photos decode much faster.
        img.draft("RGB", (IMAGE_SIZE_X, IMAGE_SIZE_Y))
        img.load()  # full decode now so bad files fail here, not later
    except UnidentifiedImageError:
        raise BadImage("Not a recognised image format")
    except (OSError, Image.DecompressionBombError) as e:
        raise BadImage(f"Could not decode image: {e}")

    img = img.convert("RGB")
    return img.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y), Image.Resampling.BICUBIC, reducing_gap=2.0)


# --- GET (health check stays HTTP)
@app.get("/")
def hello():
    return {"status": "server is live"}


# --- WebSocket (replaces POST /image)
# Protocol, per connection, strictly request -> response in order:
#   client sends: one BINARY frame = raw image file bytes (jpg/png/...)
#   server sends: one JSON text frame = same result dict as the old POST /image,
#                 or {"error": "...", "code": 400|413|500} on failure (connection stays open)
#   client may send the TEXT frame "ping" -> server replies {"type": "pong"}
@app.websocket("/ws/image")
async def image_socket(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            break

        data = msg.get("bytes")
        if data is None:
            if msg.get("text") == "ping":
                await ws.send_json({"type": "pong"})
            else:
                await ws.send_json({"error": "Send the image as a binary frame", "code": 400})
            continue

        # Backstop: uvicorn.run(ws_max_size=...) below already rejects oversized frames,
        # but that setting doesn't apply if the app is launched via the uvicorn CLI.
        if len(data) > MAX_BYTES:
            await ws.send_json({"error": "Image is too big", "code": 413})
            continue

        # Decode on the default thread pool so it overlaps with GPU work for other connections.
        try:
            img = await asyncio.to_thread(to_pil, data)
        except BadImage as e:
            await ws.send_json({"error": str(e), "code": 400})
            continue

        start_time = time.perf_counter()
        try:
            result = await loop.run_in_executor(GPU_POOL, imageDecider.infer, img)
        except Exception as e:
            await ws.send_json({"error": f"Inference failed: {e}", "code": 500})
            continue
        end_time = time.perf_counter()  # includes queue wait if other connections are busy

        print(f"Took: {(end_time - start_time) * 1000:.1f} ms")
        await ws.send_json(result)


def main() -> None:
    # Needs a websocket backend: pip install "uvicorn[standard]"  (or: pip install websockets)
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_max_size=MAX_BYTES)


if __name__ == "__main__":
    main()