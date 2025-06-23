import cv2
from PIL import Image
import numpy as np
import os
import json
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import shutil
from typing import List
import re

BUCKET_NAME = "isolated-assets"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

app = FastAPI()

@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...), style: str = Form(...)):
    validar_style(style)
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        validar_imagen(temp_path)
        imagen = cargar_imagen(temp_path)
        imagen_transp = remove_background_contiguous(imagen)
        contornos = detectar_assets_contours(imagen_transp)
        metadata = guardar_assets(imagen_transp, contornos, "output_assets", style)
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

def validar_style(style: str):
    if not re.match(r"^[A-Za-z0-9_]+$", style):
        raise HTTPException(status_code=400, detail="Style invalido")

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

def detectar_assets_grouped_contours(rgba_img: np.ndarray, area_min: int = 500, aspect_ratio_min: float = 0.1, aspect_ratio_max: float = 10.0, kernel_size: int = 7) -> list:
    """
    Detecta contornos agrupados usando dilatación sobre la máscara alfa.
    """
    alpha = rgba_img[:, :, 3]
    _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    contornos = find_grouped_contours(binary, kernel_size=kernel_size)
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

def classify_asset_type(width: int, height: int) -> str:
    ratio = width / height if height else 0
    if ratio > 2.5 or ratio < 0.4:
        return "line"
    return "square"

def guardar_assets(rgba_img: np.ndarray, contornos: list, output_dir: str, style: str):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    for i, cnt in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(cnt)
        mask = np.zeros((h, w), dtype=np.uint8)
        cnt_shifted = cnt - [x, y]
        cv2.drawContours(mask, [cnt_shifted], -1, 255, thickness=cv2.FILLED)
        asset_roi = rgba_img[y:y+h, x:x+w]
        asset_masked = asset_roi.copy()
        for c in range(3):
            asset_masked[:, :, c] = cv2.bitwise_and(asset_roi[:, :, c], asset_roi[:, :, c], mask=mask)
        asset_masked[:, :, 3] = cv2.bitwise_and(asset_roi[:, :, 3], asset_roi[:, :, 3], mask=mask)

        pil_img = Image.fromarray(cv2.cvtColor(asset_masked, cv2.COLOR_BGRA2RGBA))

        asset_type = classify_asset_type(w, h)
        if asset_type == "line" and h > w:
            pil_img = pil_img.rotate(90, expand=True)
            w, h = pil_img.size

        filename = f"{style}_{asset_type}_{i:04d}.png"
        local_path = os.path.join(output_dir, filename)
        pil_img.save(local_path)
        public_url = upload_to_supabase(local_path, filename)

        size_kb = os.path.getsize(local_path) / 1024
        metadata.append({
            "filename": filename,
            "width": w,
            "height": h,
            "size_kb": round(size_kb, 2),
            "style": style,
            "type": asset_type,
            "url": public_url,
        })

    json_path = os.path.join(output_dir, "assets_metadata.json")
    with open(json_path, "w") as jsonfile:
        json.dump(metadata, jsonfile, ensure_ascii=False, indent=2)

    upload_to_supabase(json_path, "assets_metadata.json")
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
    if filename.endswith(".json"):
        content_type = "application/json"
    elif filename.endswith(".csv"):
        content_type = "text/csv"
    else:
        content_type = "image/png"
    with open(local_path, "rb") as f:
        resp = requests.put(
            storage_url,
            headers={
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": content_type,
            },
            data=f.read(),
        )
    if resp.status_code not in (200, 201):
        raise Exception(f"Error subiendo a Supabase: {resp.status_code} {resp.text}")
    # URL pública (ajusta si tu bucket no es público)
    public_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{filename}"
    return public_url

def find_grouped_contours(
    mask: np.ndarray,
    kernel_size: int = 35  # Aumentado de 15 a 35
) -> List[np.ndarray]:
    """
    Aplica dilatación a la máscara binaria para agrupar fragmentos cercanos y luego detecta contornos.
    Args:
        mask (np.ndarray): Máscara binaria (0 y 255).
        kernel_size (int): Tamaño del kernel de dilatación.
    Returns:
        List[np.ndarray]: Lista de contornos agrupados.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=4)  # Iteraciones aumentadas a 4
    contornos, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("extractor:app", host="0.0.0.0", port=3000)
