from generative_agent.modules.scratch import Scratch
from simulation_engine.settings import *
from simulation_engine.global_methods import *
from simulation_engine.gpt_structure import *
from simulation_engine.llm_json_parser import *

def run_gpt_generate_agent(
    description,
    prompt_version="1",
    gpt_version="GPT4o",
    verbose=False):

    def create_prompt_input(description):
        return [description]

    def _func_clean_up(gpt_response, prompt=""):
        return extract_first_json_dict(gpt_response)

    def _get_fail_safe():
        return "Error Generating" ## to do, add better fail safe.

    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/create_agent/generate_agent_v1.txt"

    prompt_input = create_prompt_input(description)

    fail_safe = _get_fail_safe()

    output, prompt, prompt_input, fail_safe = chat_safe_generate(
        prompt_input, prompt_lib_file, gpt_version, 1, fail_safe,
        _func_clean_up, verbose)

    return output, [output, prompt, prompt_input, fail_safe]

def create_agent(description):
    """
    Creates a new GenerativeAgent based on a description.

    Args:
        description (str): A detailed description of the agent to be created.

    Returns:
        GenerativeAgent: A newly created GenerativeAgent instance.
    """
    agent_info, debug_info = run_gpt_generate_agent(description, "1", LLM_VERS)
    scratch = {
        "first_name": agent_info.get("first_name", ""),
        "last_name": agent_info.get("last_name", ""),
        "age": agent_info.get("age", 0),
        "sex": agent_info.get("sex", ""),
        "census_division": agent_info.get("census_division", ""),
        "political_ideology": agent_info.get("political_ideology", ""),
        "political_party": agent_info.get("political_party", ""),
        "education": agent_info.get("education", ""),
        "race": agent_info.get("race", ""),
        "ethnicity": agent_info.get("ethnicity", ""),
        "annual_income": agent_info.get("annual_income", 0.0),
        "address": agent_info.get("address", ""),
        "extraversion": agent_info.get("extraversion", 0.0),
        "agreeableness": agent_info.get("agreeableness", 0.0),
        "conscientiousness": agent_info.get("conscientiousness", 0.0),
        "neuroticism": agent_info.get("neuroticism", 0.0),
        "openness": agent_info.get("openness", 0.0),
        "fact_sheet": agent_info.get("fact_sheet", ""),
        "speech_pattern": agent_info.get("speech_pattern", ""),
        "self_description": agent_info.get("self_description", ""),
        "private_self_description": agent_info.get("private_self_description", "")
    }

    # Add initial memories if provided
    if "initial_memories" not in agent_info:
        agent_info["initial_memories"] = []

    return scratch, agent_info["initial_memories"]