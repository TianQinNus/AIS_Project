from fastapi import FastAPI

app = FastAPI(
    title="FleetCare AI API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "FleetCare AI backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }