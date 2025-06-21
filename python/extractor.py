import cv2
from PIL import Image
import numpy as np
import os
import json
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil

BUCKET_NAME = "isolated-assets"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

app = FastAPI()

@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        validar_imagen(temp_path)
        imagen = cargar_imagen(temp_path)
        imagen_transp = remove_background_contiguous(imagen)
        contornos = detectar_assets_contours(imagen_transp)
        metadata = guardar_assets(imagen_transp, contornos, "output_assets")
        return {"ok": True, "num_assets": len(metadata), "metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(temp_path)

@app.get("/")
def root():
    return {"status": "ok"}

def validar_imagen(path):
    assert os.path.exists(path), f"La ruta no existe: {path}"
    assert path.endswith((".png", ".jpg", ".jpeg")), "Formato no soportado"

def cargar_imagen(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def remove_background_contiguous(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    flood_filled = image.copy()
    for pt in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        cv2.floodFill(flood_filled, flood_mask, pt, (255, 255, 255), loDiff=(5,5,5), upDiff=(5,5,5))
    flood_gray = cv2.cvtColor(flood_filled, cv2.COLOR_BGR2GRAY)
    alpha = cv2.inRange(flood_gray, 0, 254)
    alpha_f32 = alpha.astype(np.float32) / 255.0
    alpha_f32[alpha_f32 < 0.1] = 0.0
    alpha_clean = (alpha_f32 * 255).astype(np.uint8)
    b, g, r = cv2.split(image)
    rgba = cv2.merge([b, g, r, alpha_clean])
    return rgba

def detectar_assets_contours(rgba_img: np.ndarray, area_min: int = 500) -> list:
    alpha = rgba_img[:, :, 3]
    _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contornos if cv2.contourArea(c) > area_min]

def guardar_assets(rgba_img: np.ndarray, contornos: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    for i, cnt in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(cnt)
        asset = rgba_img[y:y+h, x:x+w]
        pil_img = Image.fromarray(cv2.cvtColor(asset, cv2.COLOR_BGRA2RGBA))
        filename = f"asset_{i:03}.png"
        pil_img.save(os.path.join(output_dir, filename))
        metadata.append({"filename": filename, "bbox": [int(x), int(y), int(w), int(h)]})
    return metadata

def subir_a_supabase(local_path, remote_filename):
    """
    Sube un archivo local a Supabase Storage y retorna la URL pública.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL y SUPABASE_KEY deben estar definidos en las variables de entorno.")
    storage_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{remote_filename}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/octet-stream"
    }
    with open(local_path, "rb") as f:
        resp = requests.put(storage_url, headers=headers, data=f)
    if resp.status_code not in (200, 201):
        raise Exception(f"Error subiendo a Supabase: {resp.status_code} {resp.text}")
    # URL pública (ajusta según configuración de tu bucket)
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{remote_filename}"
    return public_url

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("extractor:app", host="0.0.0.0", port=3000)
