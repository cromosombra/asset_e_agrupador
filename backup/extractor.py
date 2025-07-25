import cv2
from PIL import Image
import numpy as np
import os
import json
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
from typing import List

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

def detectar_assets_contours(rgba_img: np.ndarray, area_min: int = 500, aspect_ratio_min: float = 0.1, aspect_ratio_max: float = 10.0) -> list:
    alpha = rgba_img[:, :, 3]
    _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h if h > 0 else 0
        if not (aspect_ratio_min <= aspect_ratio <= aspect_ratio_max):
            continue
        filtered.append(c)
    return filtered

def guardar_assets(rgba_img: np.ndarray, contornos: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    for i, cnt in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(cnt)
        # Crear máscara binaria exacta del contorno
        mask = np.zeros((h, w), dtype=np.uint8)
        cnt_shifted = cnt - [x, y]  # Ajustar contorno a ROI
        cv2.drawContours(mask, [cnt_shifted], -1, 255, thickness=cv2.FILLED)
        asset_roi = rgba_img[y:y+h, x:x+w]
        # Aplicar máscara sobre los 4 canales
        asset_masked = asset_roi.copy()
        for c in range(3):  # BGR
            asset_masked[:, :, c] = cv2.bitwise_and(asset_roi[:, :, c], asset_roi[:, :, c], mask=mask)
        asset_masked[:, :, 3] = cv2.bitwise_and(asset_roi[:, :, 3], asset_roi[:, :, 3], mask=mask)
        pil_img = Image.fromarray(cv2.cvtColor(asset_masked, cv2.COLOR_BGRA2RGBA))
        filename = f"asset_{i:03}.png"
        local_path = os.path.join(output_dir, filename)
        pil_img.save(local_path)
        # Subir a Supabase Storage
        public_url = upload_to_supabase(local_path, filename)
        metadata.append({"filename": filename, "bbox": [int(x), int(y), int(w), int(h)], "url": public_url})
    return metadata

def upload_to_supabase(local_path, filename):
    """
    Sube un archivo PNG a Supabase Storage y retorna la URL pública.
    """
    import requests
    bucket = "isolated-assets"
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    assert supabase_url and supabase_key, "SUPABASE_URL y SUPABASE_KEY deben estar configurados"
    storage_url = f"{supabase_url}/storage/v1/object/{bucket}/{filename}"
    with open(local_path, "rb") as f:
        resp = requests.put(
            storage_url,
            headers={
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "image/png"
            },
            data=f.read()
        )
    if resp.status_code not in (200, 201):
        raise Exception(f"Error subiendo a Supabase: {resp.status_code} {resp.text}")
    # URL pública (ajusta si tu bucket no es público)
    public_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{filename}"
    return public_url

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("extractor:app", host="0.0.0.0", port=3000)
