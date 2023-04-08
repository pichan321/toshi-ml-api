from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...), model: str= Form(...)):
    contents = await file.read()
    print(model)
    with open(model, "wb") as f:
        f.write(contents)
    return {"filename": file.filename, "file_size": len(contents)}