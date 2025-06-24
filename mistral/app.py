from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Mistral vLLM Service")

class TextRequest(BaseModel):
    text: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1

class TextResponse(BaseModel):
    generated_text: str
    model: str
    num_tokens: int

# Global variables for model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    
    # Using a smaller model that's more suitable for CPU
    model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
    
    try:
        print(f"Loading model: {model_name} for CPU")
        
        # Load tokenizer and model for CPU
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Using float32 for CPU
            device_map="auto",
            low_cpu_mem_usage=True  # Optimize for CPU memory
        )
        
        print("Model loaded successfully on CPU")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.get("/")
async def root():
    return {"message": "Mistral vLLM Service is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "mistral-vllm"}

@app.post("/generate", response_model=TextResponse)
async def generate_text(request: TextRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(request.text, return_tensors="pt")
        
        # Generate text with CPU-optimized settings
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=min(request.max_tokens + len(inputs.input_ids[0]), 1024),  # Cap max length
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k if request.top_k > 0 else 50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_tokens = len(outputs[0])
        
        return TextResponse(
            generated_text=generated_text,
            model="mistral-vllm",
            num_tokens=num_tokens
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
