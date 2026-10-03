import time
import io
from decider.infer import Decider
from decider.vision import VisionDecisionModel
from PIL import Image, UnidentifiedImageError, ImageOps
import torch
from pathlib import Path
from dataclasses import dataclass

from fastapi import FastAPI, File, HTTPException, UploadFile
import uvicorn

# CONFIG
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 72

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
    text: str; options: list; gold: int = 0

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

	TEMPERATURE = 1

	def __init__(self):
		self.MODEL = VisionDecisionModel("Mapika/decider-2b-vision", dtype=torch.bfloat16,
                                         grad_ckpt=False).cuda().eval()
		self.MODEL.compile(dynamic=True)

		# One example per question, so each is scored in its own row and can't influence the others
		self.EXAMPLES = [(key, opts, ImageQuestion(self.CONTEXT, [Q(text, opts, 0)])) for key, (text, opts) in self.QUESTIONS.items()]

	@torch.inference_mode()
	def infer(self, img: Image.Image) -> dict:
		inp = self.MODEL.prepare([(img, ex) for _, _, ex in self.EXAMPLES])   # 3 rows, one forward pass
		logits = self.MODEL.slot_logits(inp).float()
		assert logits.shape[0] == len(self.EXAMPLES), f"unexpected logits shape {tuple(logits.shape)}"

		answers = {}
		for row, (key, opts, _) in zip(logits, self.EXAMPLES):
			probs = torch.softmax(row.reshape(-1)[: len(opts)] / self.TEMPERATURE, dim=-1).tolist()
			answers[key] = sorted(zip(probs, opts), reverse=True)        # [(prob, option), ...]

		pos, vis, dist = answers["position"], answers["visible"], answers["distance"]
		return {
            "positions": [{"label" : label, "probs": p} for p, label in pos],
            "visibles": [{"label" : label, "probs": p} for p, label in vis],                # "yes" / "no"
            "distances": [{"label" : label, "probs": p} for p, label in dist]
        }

# Main server
app = FastAPI()
imageDecider = ImageDecider()

def to_pil(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        # JPEG only: decode at 1/2, 1/4 or 1/8 scale (never below the target). Big phone photos decode much faster.
        img.draft("RGB", (IMAGE_SIZE_X, IMAGE_SIZE_Y))
        img.load()  # full decode now so bad files fail here, not later
    except UnidentifiedImageError:
        raise HTTPException(400, "Not a recognised image format")
    except (OSError, Image.DecompressionBombError) as e:
        raise HTTPException(400, f"Could not decode image: {e}")
 
    img = img.convert("RGB")
    return img.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y), Image.Resampling.BICUBIC, reducing_gap=2.0)

# --- GET
@app.get("/")
def hello():
	return {"status" : "server is live"}

# --- POST
@app.post("/image")
def process_image(file: UploadFile = File(...)):
	data = file.file.read(MAX_BYTES + 1)
	if len(data) > MAX_BYTES:
		raise HTTPException(413, "Image is too big")
	img = to_pil(data)

	#process it futher
	start_time = time.perf_counter()
	result = imageDecider.infer(img)
	end_time = time.perf_counter()

	print("Took: ", str(end_time-start_time), "seconds")
	return result

def main() -> None:
	uvicorn.run(app, host="0.0.0.0", port=8000)