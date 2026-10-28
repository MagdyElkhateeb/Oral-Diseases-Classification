from pathlib import Path
import io
 
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from tensorflow.keras.models import load_model
 
# ---------- Settings ----------
MODEL_PATH = Path(__file__).parent / "Models" / "final_unfreeze_EfficientNetB0.keras"
IMG_SIZE = (224, 224)  # (width, height) same size used in training
CLASS_NAMES = ["Calculus", "Caries", "Gingivitis", "Mouth Ulcer", "Tooth Discoloration", "hypodontia"]
 
app = FastAPI(title="Oral Disease Classifier")
model = load_model(MODEL_PATH)
 
 
# ---------- Interface ----------
HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Oral Disease Classifier</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 480px; margin: 40px auto; padding: 0 16px; text-align: center; }
    img { max-width: 100%; max-height: 300px; margin: 16px 0; border-radius: 8px; }
    button { padding: 10px 24px; font-size: 16px; cursor: pointer; }
    .row { display: flex; align-items: center; margin: 6px 0; text-align: left; font-size: 14px; }
    .name { width: 150px; }
    .bar { flex: 1; background: #eee; border-radius: 4px; height: 14px; margin: 0 8px; }
    .fill { background: #2b6cb0; height: 100%; border-radius: 4px; }
    #result h2 { margin-bottom: 4px; }
  </style>
</head>
<body>
  <h1>Oral Disease Classifier</h1>
  <input type="file" id="file" accept="image/*"><br>
  <img id="preview" style="display:none"><br>
  <button onclick="predict()">Predict</button>
  <div id="result"></div>
 
  <script>
    const fileInput = document.getElementById("file");
    const preview = document.getElementById("preview");
 
    fileInput.onchange = () => {
      const f = fileInput.files[0];
      if (f) { preview.src = URL.createObjectURL(f); preview.style.display = "inline"; }
    };
 
    async function predict() {
      const f = fileInput.files[0];
      const out = document.getElementById("result");
      if (!f) { out.innerText = "Please choose an image first."; return; }
      out.innerText = "Predicting...";
 
      const form = new FormData();
      form.append("file", f);
      const res = await fetch("/predict", { method: "POST", body: form });
      if (!res.ok) { out.innerText = "Error: could not process the image."; return; }
      const data = await res.json();
 
      let html = `<h2>${data.prediction}</h2><p>Confidence: ${data.confidence}%</p>`;
      for (const [name, p] of Object.entries(data.probabilities)) {
        html += `<div class="row"><span class="name">${name}</span>
                 <div class="bar"><div class="fill" style="width:${p}%"></div></div>
                 <span>${p}%</span></div>`;
      }
      out.innerHTML = html;
    }
  </script>
</body>
</html>
"""
 
 
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML
 
 
# ---------- Prediction ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize(IMG_SIZE)
    x = np.expand_dims(np.array(image, dtype="float32"), axis=0)  # keep 0-255, EfficientNet normalizes inside
 
    probs = model(x, training=False).numpy()[0]
    best = int(np.argmax(probs))
 
    return {
        "prediction": CLASS_NAMES[best],
        "confidence": round(float(probs[best]) * 100, 2),
        "probabilities": {n: round(float(p) * 100, 2) for n, p in zip(CLASS_NAMES, probs)},
    }
 
 
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)