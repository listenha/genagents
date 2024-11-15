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


def _main_agent_desc(agent, anchor): 
  agent_desc = ""
  agent_desc += f"Self description: {agent.scratch.self_description}\n==\n"
  agent_desc += f"Private information: {agent.scratch.private_self_description}\n==\n"
  agent_desc += f"Other observations about the subject:\n\n"

  retrieved = agent.memory_stream.retrieve([anchor], 0, n_count=8)
  if len(retrieved) == 0:
    return agent_desc
  nodes = list(retrieved.values())[0]
  for node in nodes:
    agent_desc += f"{node.content}\n"
  return agent_desc


def _utterance_agent_desc(agent, anchor): 
  agent_desc = ""
  agent_desc += f"Self description: {agent.scratch.self_description}\n==\n"
  agent_desc += f"Speech pattern: {agent.scratch.speech_pattern}\n==\n"
  agent_desc += f"Other observations about the subject:\n\n"

  retrieved = agent.memory_stream.retrieve([anchor], 0, n_count=8)
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
  verbose=False):

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

  if len(questions) > 1: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/numerical_resp/batch_v1.txt" 
  else: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/numerical_resp/singular_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, questions, float_resp) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  if float_resp: 
    output["responses"] = [float(i) for i in output["responses"]]
  else: 
    output["responses"] = [int(i) for i in output["responses"]]

  return output, [output, prompt, prompt_input, fail_safe]


def numerical_resp(agent, questions, float_resp): 
  anchor = " ".join(list(questions.keys()))
  agent_desc = _main_agent_desc(agent, anchor)
  return run_gpt_generate_numerical_resp(
           agent_desc, questions, float_resp, "1", LLM_VERS)[0]


def run_gpt_generate_utterance(
  agent_desc, 
  str_dialogue,
  context,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False):

  def create_prompt_input(agent_desc, str_dialogue, context):
    return [agent_desc, context, str_dialogue]

  def _func_clean_up(gpt_response, prompt=""): 
    utterance = extract_first_json_dict(gpt_response)["utterance"]
    return utterance

  def _get_fail_safe():
    return None

  prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/utternace/utterance_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, str_dialogue, context) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  return output, [output, prompt, prompt_input, fail_safe]


def utterance(agent, curr_dialogue, context): 
  str_dialogue = ""
  for row in curr_dialogue:
    str_dialogue += f"[{row[0]}]: {row[1]}\n"
  str_dialogue += f"[{agent.scratch.get_fullname()}]: [Fill in]\n"

  anchor = str_dialogue
  agent_desc = _utterance_agent_desc(agent, anchor)
  return run_gpt_generate_utterance(
           agent_desc, str_dialogue, context, "1", LLM_VERS)[0]

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

def ask(agent, questions, remember, debug = False):
    # Validate and preprocess questions
    for q in questions:
        if 'response-type' not in q:
            q['response-type'] = 'open'
        if q['response-type'] == 'open' and 'response-char-limit' not in q:
            q['response-char-limit'] = 200
        if q['response-type'] == 'categorical' and 'response-options' not in q:
            raise ValueError(f"Categorical question missing response options: {q['question']}")
        if q['response-type'] in ['int', 'float'] and 'response-scale' not in q:
            raise ValueError(f"Numerical question missing response scale: {q['question']}")

    # Create anchor for memory retrieval
    anchor = " ".join([q['question'] for q in questions])
    agent_desc = _main_agent_desc(agent, anchor)

    # Generate responses
    responses, debugObj = run_gpt_generate_ask(agent_desc, questions, "1", LLM_VERS)

    # Post-process responses
    for i, q in enumerate(questions):
        if q['response-type'] == 'int':
            responses[str(i+1)]['Response'] = int(responses[str(i+1)]['Response'])
        elif q['response-type'] == 'float':
            responses[str(i+1)]['Response'] = float(responses[str(i+1)]['Response'])
    retObj = []
    for i, q in enumerate(questions):
        retObj.append({"response": responses[str(i+1)]['Response'], "reasoning": responses[str(i+1)]['Reasoning']})
    
    if debug:
        retObj.append(debugObj)

    if remember:
        for i, q in enumerate(questions):
            agent.memory_stream.remember(f"You were asked: '{q['question']}'\n You replied: '{responses[str(i+1)]['Response']}'")
      
    return retObj



  





  





  




  





  





