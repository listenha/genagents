"""
Local Model Adapter for Open-Source LLMs
Supports models via Hugging Face Transformers
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from pathlib import Path

from simulation_engine.settings import *

def get_available_gpu():
    """
    Find an available GPU with sufficient free memory.
    Returns device string like "cuda:1" or "cuda" (defaults to 0).
    """
    if not torch.cuda.is_available():
        return "cpu"
    
    import subprocess
    try:
        # Get GPU memory info from nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output: "0, 1000\n1, 2000\n..."
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                idx, free_mem = line.split(',')
                gpus.append((int(idx.strip()), int(free_mem.strip())))
        
        # Sort by free memory (descending) and find one with >10GB free
        gpus.sort(key=lambda x: x[1], reverse=True)
        for gpu_idx, free_mem_gb in gpus:
            if free_mem_gb > 10000:  # 10GB free memory threshold
                return f"cuda:{gpu_idx}"
        
        # If no GPU has 10GB free, use the one with most free memory
        if gpus:
            return f"cuda:{gpus[0][0]}"
    except Exception as e:
        print(f"Warning: Could not detect free GPU: {e}")
    
    # Fallback to default
    return "cuda"

_local_model_name = LOCAL_MODEL_NAME
_local_embedding_model = LOCAL_EMBEDDING_MODEL
_hf_token = HF_TOKEN

# If DEVICE is just "cuda", pick the GPU with the most free memory.
_device = DEVICE
if _device == "cuda":
    _device = get_available_gpu()
    print(f"Auto-selected GPU: {_device}")

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
        # Clear cache before returning to free up fragmented memory
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return _model_cache[cache_key], _tokenizer_cache[cache_key]
    
    
    # Prepare authentication
    auth_kwargs = {}
    if _hf_token:
        auth_kwargs['token'] = _hf_token
        # For older transformers versions
        auth_kwargs['use_auth_token'] = _hf_token
    
    # Local weight folders normally ship tokenizer files; if one does not,
    # fall back to the hub id of the selected model.
    model_path = Path(model_name)
    if model_path.is_dir() and not (model_path / "tokenizer.json").exists() \
            and not (model_path / "tokenizer_config.json").exists():
        print(f"Tokenizer files not found in {model_name}, loading from {LOCAL_MODEL_HF_ID}")
        tokenizer_source = LOCAL_MODEL_HF_ID
    else:
        tokenizer_source = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **auth_kwargs)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if device.startswith("cuda") and torch.cuda.is_available():
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
                       temperature: float = 0.7,
                       enable_thinking: bool = False) -> str:
    """
    Generate a response using a local model.
    
    Args:
        prompt: Input prompt text
        model_name: Model identifier (uses LOCAL_MODEL_NAME if None)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (used as-is for non-thinking mode,
                    or overridden to 0.6 for thinking mode)
        enable_thinking: Enable thinking mode (Qwen3 only, ignored for other models).
                        For Qwen3: True enables reasoning blocks, False disables them.
                        Default: False (recommended for interviews/surveys).
    
    Returns:
        Generated text response (with reasoning blocks removed if present)
    """
    try:
        model, tokenizer = load_local_model(model_name)
        device = _device
        # Use model's actual device for inputs (fixes device mismatch when using device_map="auto" or cached models, e.g. Mistral)
        model_device = next(model.parameters()).device

        # Determine actual model name for Qwen3 detection
        actual_model_name = model_name or _local_model_name
        is_qwen3 = "Qwen3" in str(actual_model_name) or "qwen3" in str(actual_model_name).lower()
        
        # Check if tokenizer supports chat templates (Qwen, Llama 3.1+, etc.)
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            # Use chat template for proper formatting
            messages = [{"role": "user", "content": prompt}]
            
            # Build template kwargs (messages must be positional, others can be keyword)
            template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": False  # Get the formatted string first
            }
            
            # Only add enable_thinking for Qwen3 models
            if is_qwen3:
                template_kwargs["enable_thinking"] = enable_thinking
            
            # Apply chat template (messages is positional argument, not keyword)
            formatted_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            
            # Now tokenize the formatted prompt
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            # Move inputs to model's device (ensures consistency with device_map="auto" / multi-device)
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # Clear cache before generation to free fragmented memory
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            # Fallback: simple prompt formatting
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # Adjust generation parameters based on thinking mode
        # Qwen3 recommendations:
        # - Thinking mode: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
        # - Non-thinking mode: Temperature=0.7, TopP=0.8, TopK=20, MinP=0
        if is_qwen3 and enable_thinking:
            gen_temperature = 0.6
            top_p = 0.95
        else:
            gen_temperature = temperature  # Use provided temperature (typically 0.7)
            top_p = 0.8
        
        # Generate with memory-efficient settings
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 512),  # Cap at 512 to save memory
                    temperature=gen_temperature,
                    top_p=top_p,
                    top_k=20,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Report memory state on OOM - convert device string to index
                    if device.startswith("cuda") and torch.cuda.is_available():
                        if ":" in device:
                            device_idx = int(device.split(":")[1])
                        else:
                            device_idx = 0
                        allocated = torch.cuda.memory_allocated(device_idx) / 1e9
                        reserved = torch.cuda.memory_reserved(device_idx) / 1e9
                        print(f"OOM Error! Memory state: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved on {device} (device_idx={device_idx})", flush=True)
                        torch.cuda.empty_cache()
                    raise
                else:
                    raise
        
        # Decode response
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clear memory after generation to free up attention matrices
        if device.startswith("cuda"):
            del inputs, outputs, generated_tokens
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clean up reasoning blocks if present (for Qwen3 thinking mode)
        # Even with enable_thinking=False, sometimes blocks may appear
        if is_qwen3:
            import re
            # Remove <think>...</think> blocks (Qwen3 thinking mode output)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
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

