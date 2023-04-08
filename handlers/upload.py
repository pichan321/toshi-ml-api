from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
router = FastAPI(default_response_class=JSONResponse)

@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    with open(file.filename, "wb") as f:
        f.write(contents)
    return {"filename": file.filename, "file_size": len(contents)}