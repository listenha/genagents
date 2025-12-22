"""
Local Model Adapter for Open-Source LLMs
Supports models via Hugging Face Transformers
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from pathlib import Path
import os

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
    # If DEVICE is just "cuda", try to find an available GPU
    if _device == "cuda":
        _device = get_available_gpu()
        print(f"Auto-selected GPU: {_device}")
except NameError:
    _device = get_available_gpu() if torch.cuda.is_available() else "cpu"

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
    # Debug: report cache status
    # if len(_model_cache) > 0:
    #     print(f"DEBUG: Model cache has {len(_model_cache)} entries: {list(_model_cache.keys())}", flush=True)
    if cache_key in _model_cache:
        # print(f"Using cached model for {cache_key}", flush=True)
        # Report memory when using cached model
        # if device.startswith("cuda") and torch.cuda.is_available():
        #     # Convert device string to device index for memory reporting
        #     if ":" in device:
        #         device_idx = int(device.split(":")[1])
        #     else:
        #         device_idx = 0
        #     allocated = torch.cuda.memory_allocated(device_idx) / 1e9
        #     reserved = torch.cuda.memory_reserved(device_idx) / 1e9
        #     # Check model dtype
        #     model = _model_cache[cache_key]
        #     first_param_dtype = next(model.parameters()).dtype
        #     print(f"Cached model dtype: {first_param_dtype}, Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved on {device}", flush=True)
        # Clear cache before returning to free up fragmented memory
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return _model_cache[cache_key], _tokenizer_cache[cache_key]
    
    # print(f"Loading new model for {cache_key} (cache miss)", flush=True)
    
    # Prepare authentication
    auth_kwargs = {}
    if _hf_token:
        auth_kwargs['token'] = _hf_token
        # For older transformers versions
        auth_kwargs['use_auth_token'] = _hf_token
    
    # Determine if we're using a local path or HuggingFace model name
    is_local_path = os.path.exists(model_name) if os.path.isabs(model_name) else False
    
    if is_local_path:
        # Check if tokenizer files exist locally (preferred - faster and guaranteed match)
        model_path = Path(model_name)
        tokenizer_json = model_path / "tokenizer.json"
        
        if tokenizer_json.exists():
            # Use local tokenizer files (best option - guaranteed to match the model)
            print(f"Using local tokenizer files from: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_kwargs)
        else:
            # Tokenizer files not found locally, try to infer HuggingFace model name
            # Path format: /path/to/models/Qwen2.5-7B-Instruct/snapshots/...
            # Extract model name from the path
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
                    if "Qwen3" in part or "Qwen2.5" in part or ("Qwen" in part and ("Instruct" in part or "32B" in part or "14B" in part or "8B" in part)):
                        tokenizer_model_name = part
                        break
            
            # Map to HuggingFace model name for downloading
            if tokenizer_model_name:
                # Handle GPTQ models (tokenizer is same as base model, just strip GPTQ suffix)
                if "GPTQ" in tokenizer_model_name:
                    # Remove GPTQ-related suffixes to get base model name
                    base_name = tokenizer_model_name.split("-GPTQ")[0]
                    tokenizer_model_name = base_name
                
                if "Qwen3" in tokenizer_model_name:
                    # Qwen3 models
                    if "32B" in tokenizer_model_name:
                        tokenizer_model_name = "Qwen/Qwen3-32B"
                    elif "14B" in tokenizer_model_name:
                        tokenizer_model_name = "Qwen/Qwen3-14B"
                    elif "8B" in tokenizer_model_name:
                        tokenizer_model_name = "Qwen/Qwen3-8B"
                    else:
                        tokenizer_model_name = "Qwen/Qwen3-32B"  # Default for Qwen3
                elif "Qwen2.5" in tokenizer_model_name:
                    # Extract version (7B, 14B, etc.)
                    if "7B" in tokenizer_model_name:
                        tokenizer_model_name = "Qwen/Qwen2.5-7B-Instruct"
                    elif "14B" in tokenizer_model_name:
                        tokenizer_model_name = "Qwen/Qwen2.5-14B-Instruct"
                    elif "32B" in tokenizer_model_name:
                        tokenizer_model_name = "Qwen/Qwen2.5-32B-Instruct"
                    else:
                        tokenizer_model_name = "Qwen/Qwen2.5-7B-Instruct"  # Default
                elif "Llama" in tokenizer_model_name:
                    tokenizer_model_name = f"meta-llama/{tokenizer_model_name}"
                else:
                    # Fallback: use the extracted name as-is
                    tokenizer_model_name = tokenizer_model_name
            else:
                # Ultimate fallback - try to infer from the full path
                # Handle GPTQ models
                if "GPTQ" in model_name:
                    # Remove GPTQ-related suffixes
                    base_name = model_name.split("-GPTQ")[0]
                    if "Qwen3" in base_name:
                        if "32B" in base_name:
                            tokenizer_model_name = "Qwen/Qwen3-32B"
                        elif "14B" in base_name:
                            tokenizer_model_name = "Qwen/Qwen3-14B"
                        elif "8B" in base_name:
                            tokenizer_model_name = "Qwen/Qwen3-8B"
                        else:
                            tokenizer_model_name = "Qwen/Qwen3-32B"
                    elif "Qwen2.5" in base_name:
                        if "32B" in base_name:
                            tokenizer_model_name = "Qwen/Qwen2.5-32B-Instruct"
                        elif "14B" in base_name:
                            tokenizer_model_name = "Qwen/Qwen2.5-14B-Instruct"
                        elif "7B" in base_name:
                            tokenizer_model_name = "Qwen/Qwen2.5-7B-Instruct"
                        else:
                            tokenizer_model_name = "Qwen/Qwen2.5-7B-Instruct"
                    else:
                        tokenizer_model_name = "Qwen/Qwen2.5-7B-Instruct"
                elif "Qwen3" in model_name:
                    if "32B" in model_name:
                        tokenizer_model_name = "Qwen/Qwen3-32B"
                    elif "14B" in model_name:
                        tokenizer_model_name = "Qwen/Qwen3-14B"
                    elif "8B" in model_name:
                        tokenizer_model_name = "Qwen/Qwen3-8B"
                    else:
                        tokenizer_model_name = "Qwen/Qwen3-32B"
                else:
                    tokenizer_model_name = "Qwen/Qwen2.5-7B-Instruct"
            
            print(f"Tokenizer files not found locally, downloading from HuggingFace: {tokenizer_model_name}")
            # Load tokenizer from HuggingFace (will download if needed)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, **auth_kwargs)
    else:
        # Use the model name directly (HuggingFace identifier)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_kwargs)
    
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
    
    # Verify model dtype and print memory usage
    # if device.startswith("cuda") and torch.cuda.is_available():
    #     # Check actual model dtype
    #     first_param_dtype = next(model.parameters()).dtype
    #     print(f"Model dtype: {first_param_dtype}, Expected: torch.float16", flush=True)
    #     
    #     # Print GPU memory usage - convert device string to index
    #     if torch.cuda.is_available():
    #         if ":" in device:
    #             device_idx = int(device.split(":")[1])
    #         else:
    #             device_idx = 0
    #         allocated = torch.cuda.memory_allocated(device_idx) / 1e9
    #         reserved = torch.cuda.memory_reserved(device_idx) / 1e9
    #         print(f"GPU memory after loading: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved on {device} (device_idx={device_idx})", flush=True)
    
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
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Clear cache before generation to free fragmented memory
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Report memory before generation - convert device string to index
                # if ":" in device:
                #     device_idx = int(device.split(":")[1])
                # else:
                #     device_idx = 0
                # allocated = torch.cuda.memory_allocated(device_idx) / 1e9
                # reserved = torch.cuda.memory_reserved(device_idx) / 1e9
                # print(f"Memory before generation: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved on {device} (device_idx={device_idx})", flush=True)
        else:
            # Fallback: simple prompt formatting
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
            # Report memory after generation
            # if ":" in device:
            #     device_idx = int(device.split(":")[1])
            # else:
            #     device_idx = 0
            # allocated = torch.cuda.memory_allocated(device_idx) / 1e9
            # reserved = torch.cuda.memory_reserved(device_idx) / 1e9
            # print(f"Memory after generation: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved on {device}", flush=True)
        
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

