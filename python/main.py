import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from python.extractor import (
    validar_imagen, cargar_imagen, quitar_fondo_negro,
    detectar_contornos, recortar_assets
)
import shutil

app = FastAPI()

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    # Guardar archivo temporalmente
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        validar_imagen(temp_path)
        imagen = cargar_imagen(temp_path)
        imagen_transp = quitar_fondo_negro(imagen)
        contornos = detectar_contornos(imagen_transp)
        metadata = recortar_assets(imagen_transp, contornos, "output_assets")
        return JSONResponse({"ok": True, "num_assets": len(metadata), "metadata": metadata})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(temp_path)

@app.get("/")
def root():
    return {"status": "ok"}
