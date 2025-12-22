import math
import sys
import datetime
import random
import string
import re

from numpy import dot
from numpy.linalg import norm

from simulation_engine.settings import * 
from simulation_engine.global_methods import *
from simulation_engine.gpt_structure import *
from simulation_engine.llm_json_parser import *


def _main_agent_desc(agent, anchor, n_count=None, time_step=None): 
  agent_desc = ""
  agent_desc += f"Self description: {agent.get_self_description()}\n==\n"
  agent_desc += f"Other observations about the subject:\n\n"

  # Use global defaults if not provided
  if n_count is None:
    n_count = MEMORY_N_COUNT
  if time_step is None:
    time_step = MEMORY_TIME_STEP

  retrieved = agent.memory_stream.retrieve([anchor], time_step, n_count=n_count)
  if len(retrieved) == 0:
    return agent_desc
  nodes = list(retrieved.values())[0]
  for node in nodes:
    agent_desc += f"{node.content}\n"
  return agent_desc


def _utterance_agent_desc(agent, anchor, n_count=None, time_step=None, curr_filter="reflection"): 
  agent_desc = ""
  agent_desc += f"Self description: {agent.get_self_description()}\n==\n"
  agent_desc += f"Other observations about the subject:\n\n"

  # Use global defaults if not provided
  if n_count is None:
    n_count = MEMORY_N_COUNT
  if time_step is None:
    time_step = MEMORY_TIME_STEP

  # Retrieve only reflection nodes to avoid dialogue repetition (observation nodes contain Q&A which is already in dialogue history)
  retrieved = agent.memory_stream.retrieve([anchor], time_step, n_count=n_count, curr_filter=curr_filter)
  if len(retrieved) == 0:
    return agent_desc
  
  nodes = list(retrieved.values())[0]
  for node in nodes:
    agent_desc += f"{node.content}\n"
  return agent_desc


def run_gpt_generate_categorical_resp(
  agent_desc, 
  questions,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False):

  def create_prompt_input(agent_desc, questions):
    str_questions = ""
    for key, val in questions.items(): 
      str_questions += f"Q: {key}\n"
      str_questions += f"Option: {val}\n\n"
    str_questions = str_questions.strip()
    return [agent_desc, str_questions]

  def _func_clean_up(gpt_response, prompt=""): 
    responses, reasonings = extract_first_json_dict_categorical(gpt_response)
    ret = {"responses": responses, "reasonings": reasonings}
    return ret

  def _get_fail_safe():
    return None

  if len(questions) > 1: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/batch_v1.txt" 
  else: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/singular_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, questions) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  return output, [output, prompt, prompt_input, fail_safe]


def categorical_resp(agent, questions): 
  anchor = " ".join(list(questions.keys()))
  agent_desc = _main_agent_desc(agent, anchor)
  return run_gpt_generate_categorical_resp(
           agent_desc, questions, "1", LLM_VERS)[0]


def run_gpt_generate_numerical_resp(
  agent_desc, 
  questions, 
  float_resp,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False,
  prompt_template=None):

  def create_prompt_input(agent_desc, questions, float_resp):
    str_questions = ""
    for key, val in questions.items(): 
      str_questions += f"Q: {key}\n"
      str_questions += f"Range: {str(val)}\n\n"
    str_questions = str_questions.strip()

    if float_resp: 
      resp_type = "float"
    else: 
      resp_type = "integer"
    return [agent_desc, str_questions, resp_type]

  def _func_clean_up(gpt_response, prompt=""): 
    responses, reasonings = extract_first_json_dict_numerical(gpt_response)
    ret = {"responses": responses, "reasonings": reasonings}
    return ret

  def _get_fail_safe():
    return None

  # Use provided template or default to batch/singular based on question count
  if prompt_template is None:
    if len(questions) > 1: 
      prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/numerical_resp/batch_v1.txt" 
    else: 
      prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/numerical_resp/singular_v1.txt"
  else:
    # If prompt_template is provided, use it directly (should be full path or relative to LLM_PROMPT_DIR)
    if prompt_template.startswith("/") or prompt_template.startswith("."):
      prompt_lib_file = prompt_template
    else:
      # Assume it's in numerical_resp directory
      prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/numerical_resp/{prompt_template}"

  prompt_input = create_prompt_input(agent_desc, questions, float_resp) 
  fail_safe = _get_fail_safe() 

  # For batch questions, increase max_tokens to avoid truncation
  # Each question needs ~200-300 tokens for reasoning + response
  num_questions = len(questions)
  if num_questions > 1:
    # Calculate max_tokens: ~300 per question + buffer
    calculated_max_tokens = max(2000, num_questions * 300)
  else:
    calculated_max_tokens = 1500  # Default for single question

  output, prompt, prompt_input, fail_safe = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose, max_tokens=calculated_max_tokens)

  if float_resp: 
    output["responses"] = [float(i) for i in output["responses"]]
  else: 
    output["responses"] = [int(i) for i in output["responses"]]

  return output, [output, prompt, prompt_input, fail_safe]


def numerical_resp(agent, questions, float_resp, prompt_template=None): 
  anchor = " ".join(list(questions.keys()))
  agent_desc = _main_agent_desc(agent, anchor)
  return run_gpt_generate_numerical_resp(
           agent_desc, questions, float_resp, "1", LLM_VERS, 
           verbose=False, prompt_template=prompt_template)[0]


def run_gpt_generate_utterance(
  agent_desc, 
  str_dialogue,
  context,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False,
  prompt_template=None):

  def create_prompt_input(agent_desc, str_dialogue, context):
    # For interview template, context is embedded in the template itself
    # For other templates (like utterance_v1.txt), context is passed as INPUT 2
    # Always return 3 elements for compatibility, but context may be empty for interview template
    return [agent_desc, str_dialogue, context]

  def _func_clean_up(gpt_response, prompt=""): 
    result = extract_first_json_dict(gpt_response)
    if result is None:
      # If JSON parsing fails, try to extract the utterance directly from the response
      # Look for common patterns in the response
      response_lower = gpt_response.lower()
      if "utterance" in response_lower or "response" in response_lower:
        # Try to find text after "utterance" or similar keywords
        import re
        # Look for JSON-like structure even if not perfect
        utterance_match = re.search(r'["\']utterance["\']\s*:\s*["\']([^"\']+)["\']', gpt_response, re.IGNORECASE)
        if utterance_match:
          return utterance_match.group(1)
        # Or just return the response if it looks like a direct answer
        if len(gpt_response.strip()) > 10 and not gpt_response.strip().startswith('{'):
          return gpt_response.strip()
      # Last resort: return a message indicating parsing failed
      return f"[Response received but JSON parsing failed. Raw response: {gpt_response[:200]}...]"
    return result.get("utterance", gpt_response.strip())

  def _get_fail_safe():
    return None

  # Use provided template or default to utterance_v1.txt
  if prompt_template is None:
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/utternace/utterance_v1.txt" 
  else:
    prompt_lib_file = prompt_template 

  prompt_input = create_prompt_input(agent_desc, str_dialogue, context) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  return output, [output, prompt, prompt_input, fail_safe]


def utterance(agent, curr_dialogue, context="", prompt_template=None): 
  str_dialogue = ""
  for row in curr_dialogue:
    str_dialogue += f"[{row[0]}]: {row[1]}\n"
  str_dialogue += f"[{agent.get_fullname()}]: [Fill in]\n"

  anchor = str_dialogue
  # For interview template, retrieve only reflection nodes to avoid dialogue repetition
  # For other templates, use default (all nodes)
  curr_filter = "reflection" if prompt_template and "interview" in prompt_template else "all"
  agent_desc = _utterance_agent_desc(agent, anchor, curr_filter=curr_filter)
  return run_gpt_generate_utterance(
           agent_desc, str_dialogue, context, "1", LLM_VERS, 
           verbose=False, prompt_template=prompt_template)[0]

##  Ask function.
def run_gpt_generate_ask(
    agent_desc,
    questions,
    prompt_version="1",
    gpt_version="GPT4o",
    verbose=False):

    def create_prompt_input(agent_desc, questions):
        str_questions = ""
        i = 1
        for q in questions:
            str_questions += f"Q{i}: {q['question']}\n"
            str_questions += f"Type: {q['response-type']}\n"
            if q['response-type'] == 'categorical':
                str_questions += f"Options: {', '.join(q['response-options'])}\n"
            elif q['response-type'] in ['int', 'float']:
                str_questions += f"Range: {q['response-scale']}\n"
            elif q['response-type'] == 'open':
                char_limit = q.get('response-char-limit', 200)
                str_questions += f"Character Limit: {char_limit}\n"
            str_questions += "\n"
            i += 1
        return [agent_desc, str_questions.strip()]

    def _func_clean_up(gpt_response, prompt=""):
        responses = extract_first_json_dict(gpt_response)
        return responses

    def _get_fail_safe():
        return None

    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/ask/batch_v1.txt"

    prompt_input = create_prompt_input(agent_desc, questions)
    fail_safe = _get_fail_safe()

    output, prompt, prompt_input, fail_safe = chat_safe_generate(
        prompt_input, prompt_lib_file, gpt_version, 1, fail_safe,
        _func_clean_up, verbose)

    return output, [output, prompt, prompt_input, fail_safe]



  





  




  





  





