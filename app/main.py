from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import io

import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.EAMNet import Network


app = FastAPI()


# =========================
# 1. 模型配置
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    PROJECT_ROOT
    / "snapshot"
    / "EAMNet_E_b4_3"
    / "Net_epoch_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
RESULT_DIR = PROJECT_ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

# =========================
# 2. 加载 EAMNet
# =========================

model = Network(imagenet_pretrained=False)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()
transform = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# 3. API
# =========================

@app.get("/")
def home():
    return {
        "message": "EAMNet API",
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

    main_prediction = outputs[1]

    prediction = torch.sigmoid(main_prediction)

    prediction = prediction.cpu().numpy().squeeze()

    prediction = (
        prediction * 255
    ).clip(0, 255).astype("uint8")

    result_path = (
        RESULT_DIR
        / f"{Path(file.filename).stem}_prediction.png"
    )

    Image.fromarray(prediction).save(result_path)

    return {
        "success": True,
        "filename": file.filename,
        "output_count": len(outputs),
        "prediction_shape": list(main_prediction.shape),
        "result_path": str(result_path),
        "message": "EAMNet prediction completed"
    }