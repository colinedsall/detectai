from fastapi import FastAPI
from app.routers import text

app = FastAPI(title="detectai", version="0.1.0")

# Health
@app.get("/health")
async def health():
    return {"status": "ok"}

# Routers
app.include_router(text.router, prefix="/v1/detect", tags=["detect"])
