"""
Local Model Adapter for Open-Source LLMs
Supports models via Hugging Face Transformers
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import os

from simulation_engine.settings import *

# Get local model settings with defaults
try:
    _local_model_name = LOCAL_MODEL_NAME
except NameError:
    _local_model_name = "Qwen/Qwen2.5-7B-Instruct"

try:
    _local_embedding_model = LOCAL_EMBEDDING_MODEL
except NameError:
    _local_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

try:
    _device = DEVICE
except NameError:
    _device = "cuda"

try:
    _hf_token = HF_TOKEN
except NameError:
    _hf_token = None

# Global model cache to avoid reloading
_model_cache = {}
_tokenizer_cache = {}
_embedding_model_cache = None


def load_local_model(model_name: str = None, device: str = None):
    """
    Load a local model and tokenizer, with caching to avoid reloading.
    
    Args:
        model_name: Hugging Face model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
        device: Device to load model on ("cuda", "cpu", "mps")
    
    Returns:
        tuple: (model, tokenizer)
    """
    if model_name is None:
        model_name = _local_model_name
    if device is None:
        device = _device
    
    # Check cache first
    cache_key = f"{model_name}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key], _tokenizer_cache[cache_key]
    
    print(f"Loading model: {model_name} on {device}")
    
    # Prepare authentication
    auth_kwargs = {}
    if _hf_token:
        auth_kwargs['token'] = _hf_token
        # For older transformers versions
        auth_kwargs['use_auth_token'] = _hf_token
    
    # Determine if we're using a local path or HuggingFace model name
    # If it's a local path, use the model name for tokenizer (which will download it)
    # but use local path for model weights
    is_local_path = os.path.exists(model_name) if os.path.isabs(model_name) else False
    
    if is_local_path:
        # For local paths, try to infer the model name from the path
        # Path format: /path/to/models/Qwen2.5-7B-Instruct/snapshots/...
        # Extract "Qwen2.5-7B-Instruct" from the path
        path_parts = model_name.split('/')
        
        # Find the model directory name (usually before "snapshots")
        tokenizer_model_name = None
        for i, part in enumerate(path_parts):
            if part == "snapshots" and i > 0:
                # The model name should be the directory before "snapshots"
                tokenizer_model_name = path_parts[i-1]
                break
        
        # If we didn't find it, try to extract from the path
        if not tokenizer_model_name:
            # Look for common model name patterns in the path
            for part in path_parts:
                if "Qwen2.5" in part or ("Qwen" in part and "Instruct" in part):
                    tokenizer_model_name = part
                    break
        
        # Map to HuggingFace model name
        if tokenizer_model_name:
            if "Qwen2.5" in tokenizer_model_name:
                # Extract version (7B, 14B, etc.)
                if "7B" in tokenizer_model_name:
                    tokenizer_model_name = "Qwen/Qwen2.5-7B-Instruct"
                elif "14B" in tokenizer_model_name:
                    tokenizer_model_name = "Qwen/Qwen2.5-14B-Instruct"
                else:
                    tokenizer_model_name = "Qwen/Qwen2.5-7B-Instruct"  # Default
            elif "Llama" in tokenizer_model_name:
                tokenizer_model_name = f"meta-llama/{tokenizer_model_name}"
            else:
                # Fallback: use the extracted name as-is
                tokenizer_model_name = tokenizer_model_name
        else:
            # Ultimate fallback
            tokenizer_model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        print(f"Using HuggingFace model name '{tokenizer_model_name}' for tokenizer (local path for model weights)")
        # Load tokenizer from HuggingFace (will download if needed)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, **auth_kwargs)
    else:
        # Use the model name directly (HuggingFace identifier)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_kwargs)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if device == "cuda" and torch.cuda.is_available():
        try:
            # Try using device_map for automatic GPU placement
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
                **auth_kwargs
            )
        except (ValueError, ImportError):
            # Fallback: load to GPU manually
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                **auth_kwargs
            )
            model = model.to(device)
    else:
        # CPU mode
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            **auth_kwargs
        )
        if device != "cpu":
            model = model.to(device)
    
    model.eval()
    
    # Cache the model
    _model_cache[cache_key] = model
    _tokenizer_cache[cache_key] = tokenizer
    
    print(f"Model loaded successfully!")
    return model, tokenizer


def load_embedding_model(model_name: str = None):
    """
    Load a sentence transformer model for embeddings.
    
    Args:
        model_name: Sentence transformer model name
    
    Returns:
        SentenceTransformer model
    """
    global _embedding_model_cache
    
    if model_name is None:
        model_name = _local_embedding_model
    
    if _embedding_model_cache is None:
        print(f"Loading embedding model: {model_name}")
        _embedding_model_cache = SentenceTransformer(model_name)
        print("Embedding model loaded!")
    
    return _embedding_model_cache


def local_model_request(prompt: str, 
                       model_name: str = None,
                       max_tokens: int = 1500,
                       temperature: float = 0.7) -> str:
    """
    Generate a response using a local model.
    
    Args:
        prompt: Input prompt text
        model_name: Model identifier (uses LOCAL_MODEL_NAME if None)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text response
    """
    try:
        model, tokenizer = load_local_model(model_name)
        device = _device
        
        # Check if tokenizer supports chat templates (Qwen, Llama 3.1+, etc.)
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            # Use chat template for proper formatting
            messages = [{"role": "user", "content": prompt}]
            # Apply chat template and tokenize
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False  # Get the formatted string first
            )
            # Now tokenize the formatted prompt
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            # Fallback: simple prompt formatting
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        input_length = inputs["input_ids"].shape[1]
        
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Generation error details:\n{error_details}")
        return f"GENERATION ERROR: {str(e)}"


def local_get_text_embedding(text: str, model_name: str = None) -> List[float]:
    """
    Generate text embedding using a local sentence transformer model.
    
    Args:
        text: Input text
        model_name: Embedding model name (uses LOCAL_EMBEDDING_MODEL if None)
    
    Returns:
        List of embedding values
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    
    text = text.replace("\n", " ").strip()
    model = load_embedding_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True).tolist()
    
    return embedding

