import openai
import time
from typing import List, Union

from simulation_engine.settings import *

if MODEL_PROVIDER == "local":
  from simulation_engine.local_model_adapter import (
    local_model_request,
    local_get_text_embedding
  )
else:
  openai.api_key = OPENAI_API_KEY


# ============================================================================
# #######################[SECTION 1: HELPER FUNCTIONS] #######################
# ============================================================================

def print_run_prompts(prompt_input: Union[str, List[str]], 
                      prompt: str, 
                      output: str) -> None:
  print (f"=== START =======================================================")
  print ("~~~ prompt_input    ----------------------------------------------")
  print (prompt_input, "\n")
  print ("~~~ prompt    ----------------------------------------------------")
  print (prompt, "\n")
  print ("~~~ output    ----------------------------------------------------")
  print (output, "\n") 
  print ("=== END ==========================================================")
  print ("\n\n\n")


def generate_prompt(prompt_input: Union[str, List[str]], 
                    prompt_lib_file: str) -> str:
  """Generate a prompt by replacing placeholders in a template file with 
     input."""
  if isinstance(prompt_input, str):
    prompt_input = [prompt_input]
  prompt_input = [str(i) for i in prompt_input]

  with open(prompt_lib_file, "r") as f:
    prompt = f.read()

  for count, input_text in enumerate(prompt_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", input_text)

  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

  return prompt.strip()


# ============================================================================
# ####################### [SECTION 2: SAFE GENERATE] #########################
# ============================================================================

def gpt_request(prompt: str, 
                model: str = "gpt-4o", 
                max_tokens: int = 1500) -> str:
  """Make a request to either OpenAI or local model based on MODEL_PROVIDER."""
  if MODEL_PROVIDER == "local":
    return local_model_request(prompt, model_name=LOCAL_MODEL_NAME,
                               max_tokens=max_tokens, temperature=0.7)
  else:
    if model == "o1-preview": 
      try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
          model=model,
          messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
      except Exception as e:
        return f"GENERATION ERROR: {str(e)}"

    try:
      client = openai.OpenAI(api_key=OPENAI_API_KEY)
      # GPT-5 family uses max_completion_tokens instead of max_tokens.
      kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
      }
      if str(model).startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_tokens
      else:
        kwargs["max_tokens"] = max_tokens

      response = client.chat.completions.create(**kwargs)
      return response.choices[0].message.content
    except Exception as e:
      return f"GENERATION ERROR: {str(e)}"


def chat_safe_generate(prompt_input: Union[str, List[str]], 
                       prompt_lib_file: str,
                       gpt_version: str = "gpt-4o", 
                       repeat: int = 1,
                       fail_safe: str = "error", 
                       func_clean_up: callable = None,
                       verbose: bool = False,
                       max_tokens: int = 1500) -> tuple:
  """Generate a response using GPT models with error handling & retries."""
  prompt = generate_prompt(prompt_input, prompt_lib_file)
  for i in range(repeat):
    response = gpt_request(prompt, model=gpt_version, max_tokens=max_tokens)
    if response != "GENERATION ERROR":
      break
    time.sleep(2**i)
  else:
    response = fail_safe

  if func_clean_up:
    response = func_clean_up(response, prompt=prompt)

  if verbose or DEBUG:
    print_run_prompts(prompt_input, prompt, response)

  return response, prompt, prompt_input, fail_safe


# ============================================================================
# #################### [SECTION 3: OTHER API FUNCTIONS] ######################
# ============================================================================

def get_text_embedding(text: str, 
                       model: str = "text-embedding-3-small") -> List[float]:
  """Generate embedding using either OpenAI or local model."""
  if MODEL_PROVIDER == "local":
    return local_get_text_embedding(text, model_name=LOCAL_EMBEDDING_MODEL)
  else:
    if not isinstance(text, str) or not text.strip():
      raise ValueError("Input text must be a non-empty string.")

    text = text.replace("\n", " ").strip()
    response = openai.embeddings.create(
      input=[text], model=model).data[0].embedding
    return response
