from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
router = APIRouter()

@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...), model: str= Form(...)):
    contents = await file.read()

    with open(model, "wb") as f:
        f.write(contents)

    columns = list(pd.read_csv(model).columns)

    return {"filename": file.filename, "file_size": len(contents), "columns": columns}