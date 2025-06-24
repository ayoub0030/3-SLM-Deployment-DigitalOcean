from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

app = FastAPI(title="Qwen SLM Service")

class TextRequest(BaseModel):
    text: str
    max_length: int = 100

class TextResponse(BaseModel):
    generated_text: str
    model: str

# Global variables for model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B")
    
    try:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.get("/")
async def root():
    return {"message": "Qwen SLM Service is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "qwen"}

@app.post("/generate", response_model=TextResponse)
async def generate_text(request: TextRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer(request.text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=request.max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return TextResponse(
            generated_text=generated_text,
            model="qwen"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))