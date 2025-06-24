from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from vllm import LLM, SamplingParams

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
llm = None
sampling_params = SamplingParams()

@app.on_event("startup")
async def startup_event():
    global llm, sampling_params
    
    model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
    
    try:
        print(f"Loading model: {model_name} with vLLM")
        
        # Initialize vLLM LLM instance
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Adjust based on available GPUs
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        
        # Set default sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=1.0,
            max_tokens=100,
        )
        
        print("Model loaded successfully with vLLM")
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
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Update sampling params with request parameters
        current_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            top_k=request.top_k if request.top_k > 0 else -1
        )
        
        # Generate text using vLLM
        outputs = llm.generate([request.text], current_params)
        
        # Extract the generated text from the output
        generated_text = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)
        
        return TextResponse(
            generated_text=generated_text,
            model="mistral-vllm",
            num_tokens=num_tokens
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
