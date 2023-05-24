from typing import Union
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

#router imports
from handlers.upload import router as upload_router
from handlers.model import router as model_router
from handlers.training import router as training_router

app = FastAPI(default_response_class=JSONResponse)

#middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

#routers
app.include_router(upload_router)
app.include_router(model_router)
app.include_router(training_router)







