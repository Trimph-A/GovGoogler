from fastapi import APIRouter, HTTPException, FastAPI
from api.routes import router


app = FastAPI(
    title="DMV Document Validator and Assistant",
    description="A combined API for document validation and DMV assistance.",
    version="1.0.0",
)
# Configuration for xAI
# XAI_API_KEY = 'ai-ETigqEp3O9hficmCWrOd414VuQzBoJDxizaWjJcWLxnwYtj9DsAAUaycwMZpbpb4Ohe0ltceO3DaTeCn'

# Rejestracja tras
app.include_router(router)
print(app.routes)  # This will display all registered routes

@app.get("/")
def read_root():
    return {"message": "Welcome to the DMV Document Validator and Assistant API"}


