import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from python.extractor import (
    validar_imagen, cargar_imagen, remove_background_contiguous,
    detectar_assets_grouped_contours, guardar_assets, validar_style
)
import shutil

app = FastAPI()

# Ensure output_assets directory exists at startup
os.makedirs("output_assets", exist_ok=True)

# Serve static files from the output_assets directory
app.mount("/assets", StaticFiles(directory="output_assets"), name="assets")

@app.post("/extract")
async def extract(file: UploadFile = File(...), style: str = Form(...)):
    validar_style(style)
    # Guardar archivo temporalmente
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        validar_imagen(temp_path)
        imagen = cargar_imagen(temp_path)
        imagen_transp = remove_background_contiguous(imagen)
        contornos = detectar_assets_grouped_contours(imagen_transp)
        metadata = guardar_assets(imagen_transp, contornos, "output_assets", style)
        return JSONResponse({"ok": True, "num_assets": len(metadata), "metadata": metadata})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(temp_path)

@app.get("/")
def root():
    return {"status": "ok"}
