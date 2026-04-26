from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

app = FastAPI(title="Weather Analysis API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/upload/")
async def upload_dataset(file: UploadFile = File(...)):
    contents = await file.read()
  
    df = pd.read_csv(io.BytesIO(contents))
    
    return {
        "filename": file.filename,
        "message": "File uploaded successfully",
        "rows": df.shape[0],
        "columns": df.shape[1],
        "columns_list": df.columns.tolist()
    }