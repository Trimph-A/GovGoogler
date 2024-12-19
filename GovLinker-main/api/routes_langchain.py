from fastapi import APIRouter, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from utils.image_utils import encode_image_to_base64, convert_pdf_to_images, pil_image_to_base64
import os
from langchain_xai import ChatXAI
from openai import OpenAI
from utils.utils import preprocess_text
from api.scrapping import scrape_html, scrape_pdf
from api.embeddings import embed_and_store_data
from api.faiss_index import search_faiss
from langchain_xai import ChatXAI

# Configuration
XAI_API_KEY = 'xai-ETigqEp3O9hficmCWrOd414VuQzBoJDxizaWjJcWLxnwYtj9DsAAUaycwMZpbpb4Ohe0ltceO3DaTeCn' 
print(f"Loaded API Key: {XAI_API_KEY}")
VISION_MODEL_NAME = "grok-vision-beta"
CHAT_MODEL_NAME = "grok-beta"


# Initialize ChatXAI client
client = OpenAI(
    model=CHAT_MODEL_NAME,
    api_key=XAI_API_KEY
)

router = APIRouter()

# Mocked documents
DOCUMENTS_DB = {
    "driver_license_application": {
        "document_name": "Driver's License Application Form",
        "url": "https://www.dmv.virginia.gov/licenses-ids/license/applying"
    },
    "id_card_application": {
        "document_name": "State ID Application Form",
        "url": "https://dds.georgia.gov/georgia-licenseid/new-licenseid/apply-new-ga-license"
    },
    "vehicle_registration": {
        "document_name": "Vehicle Registration Form",
        "url": "https://www.usa.gov/state-motor-vehicle-services"
    },
}

class DocumentCheckResult(BaseModel):
    is_valid: bool
    missing_fields: List[str]
    errors: List[str] 

class QuestionRequest(BaseModel):
    question: str

class DocumentRequest(BaseModel):
    document_type: str

class DocumentResponse(BaseModel):
    document_name: str
    url: str

@router.post("/validate-document", response_model=DocumentCheckResult)
async def validate_document(file: UploadFile):
    if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only JPEG, PNG, and PDF are allowed.")

    if file.content_type == "application/pdf":
        images = convert_pdf_to_images(file.file)
        base64_images = [pil_image_to_base64(image) for image in images]
    else:
        base64_image = encode_image_to_base64(file.file)
        base64_images = [base64_image]

    results = []
    for image in base64_images:
        result = process_image_with_grok(image)
        results.append(result)

    aggregated_result = analyze_document_results(results)
    return aggregated_result

def process_image_with_grok(base64_image: str) -> dict:
    response = client.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract and validate all fields in this document match to headlines?",
                    },
                ],
            }
        ],
    )
    return response.choices[0].message

def analyze_document_results(results: List[dict]) -> DocumentCheckResult:
    required_fields = ["Name", "Date of Birth", "Document Number", "Expiration Date"]
    missing_fields = []
    errors = []
    for field in required_fields:
        if not any(field in result.get("content", "") for result in results):
            missing_fields.append(field)
    is_valid = len(missing_fields) == 0
    return DocumentCheckResult(is_valid=is_valid, missing_fields=missing_fields, errors=errors)

@router.post("/generate-response", response_model=List[str])
def ask_question(request: QuestionRequest):
    messages = [
        {"role": "system", "content": "You are a helpful assistant for DMV-related processes and documents."},
        {"role": "user", "content": request.question}
    ]

    response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=messages
        )
    return [response]

def process_chat_with_grok(messages: List[dict]) -> str:
    response = client.send_messages(messages=messages)  # Przykład innej metody
    return response.choices[0].message["content"]


@router.post("/get-document", response_model=DocumentResponse)
def get_document_endpoint(request: DocumentRequest):
    document = DOCUMENTS_DB.get(request.document_type)
    if not document:
        raise HTTPException(status_code=404, detail="Document type not found")
    return DocumentResponse(**document)

# Request model for query
class QueryRequest(BaseModel):
    query: str

# POST endpoint for querying
@router.post("/ask")
async def ask(query_request: QueryRequest):
    query = query_request.query
    try:
        # Query FAISS for relevant documents
        relevant_docs = search_faiss(query)
        
        if not relevant_docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        
        # Query xAI for the generated response
        response = ask_xai(relevant_docs, query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to use xAI (OpenAI in this case)
def ask_xai(relevant_docs, query):
    try:
        client = ChatXAI(api_key=XAI_API_KEY)
        # Format the documents into a string or appropriate format for xAI
        formatted_docs = "\n".join([doc['document_name'] for doc in relevant_docs])  # Example formatting
        response = client.query(input_data=formatted_docs, question=query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"xAI query failed: {str(e)}")
