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
        imagen_transp = quitar_fondo_negro(imagen)
        contornos = detectar_contornos(imagen_transp)
        metadata = recortar_assets(imagen_transp, contornos, "output_assets")
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

def quitar_fondo_negro(imagen_cv):
    gray = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(imagen_cv)
    rgba = cv2.merge([b, g, r, alpha])
    return rgba

def detectar_contornos(imagen_rgba):
    gray = cv2.cvtColor(imagen_rgba, cv2.COLOR_BGRA2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

def recortar_assets(imagen_rgba, contornos, carpeta_salida):
    os.makedirs(carpeta_salida, exist_ok=True)
    metadata = []

    for i, c in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(c)
        if w < 20 or h < 20:
            continue
        asset = imagen_rgba[y:y+h, x:x+w]
        filename = f"asset_{i:02d}.png"
        path = os.path.join(carpeta_salida, filename)
        cv2.imwrite(path, asset)

        metadata.append({
            "index": i,
            "filename": filename,
            "position": {"x": x, "y": y},
            "dimensions": {"width": w, "height": h}
        })

    with open(os.path.join(carpeta_salida, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

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
