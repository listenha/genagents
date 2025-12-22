import uuid

from genagents.modules.interaction import *
from genagents.modules.memory_stream import *


# ############################################################################
# ###                        GENERATIVE AGENT CLASS                        ###
# ############################################################################

class GenerativeAgent: 
  def __init__(self, agent_folder=None):
    if agent_folder: 
      # We stop the process if the agent storage folder already exists. 
      if not check_if_file_exists(f"{agent_folder}/scratch.json"):
        print ("Generative agent does not exist in the current location.")
        return 
      
      # Loading the agent's memories. 
      with open(f"{agent_folder}/scratch.json") as json_file:
        scratch = json.load(json_file)
      with open(f"{agent_folder}/memory_stream/embeddings.json") as json_file:
        embeddings = json.load(json_file)
      with open(f"{agent_folder}/memory_stream/nodes.json") as json_file:
        nodes = json.load(json_file)

      self.id = uuid.uuid4()
      self.scratch = scratch
      self.memory_stream = MemoryStream(nodes, embeddings)

      # Check for embedding dimension mismatch and re-embed if needed
      self._check_and_fix_embeddings(agent_folder)

    else: 
      self.id = uuid.uuid4()
      self.scratch = {}
      self.memory_stream = MemoryStream([], {})


  def update_scratch(self, update): 
    self.scratch.update(update)
      

  def package(self): 
    """
    Packaging the agent's meta info for saving. 

    Parameters:
      None
    Returns: 
      packaged dictionary
    """
    return {"id": str(self.id)}


  def save(self, save_directory): 
    """
    Given a save_code, save the agents' state in the storage. Right now, the 
    save directory works as follows: 
    'storage/<agent_name>/<save_code>'

    As you grow different versions of the agent, save the new agent state in 
    a different save code location. Remember that 'init' is the originally
    initialized agent directory.

    Parameters:
      save_code: str
    Returns: 
      None
    """
    # Name of the agent and the current save location. 
    storage = save_directory
    create_folder_if_not_there(f"{storage}/memory_stream")
    
    # Saving the agent's memory stream. This includes saving the embeddings 
    # as well as the nodes. 
    with open(f"{storage}/memory_stream/embeddings.json", "w") as json_file:
      json.dump(self.memory_stream.embeddings, 
                json_file)
    with open(f"{storage}/memory_stream/nodes.json", "w") as json_file:
      json.dump([node.package() for node in self.memory_stream.seq_nodes], 
                json_file, indent=2)

    # Saving the agent's scratch memories. 
    with open(f"{storage}/scratch.json", "w") as json_file:
      json.dump(self.scratch, json_file, indent=2)

    # Saving the agent's meta information. 
    with open(f"{storage}/meta.json", "w") as json_file:
      json.dump(self.package(), json_file, indent=2)


  def get_fullname(self): 
    if "first_name" in self.scratch and "last_name" in self.scratch:
      return f"{self.scratch['first_name']} {self.scratch['last_name']}"
    else: 
      return ""

  def get_self_description(self): 
    return str(self.scratch)

  def remember(self, content, time_step=0): 
    """
    Add a new observation to the memory stream. 

    Parameters:
      content: The content of the current memory record that we are adding to
        the agent's memory stream. 
    Returns: 
      None
    """
    self.memory_stream.remember(content, time_step)


  def reflect(self, anchor, time_step=0, reflection_count=None): 
    """
    Add a new reflection to the memory stream. 

    Parameters:
      anchor: str reflection anchor
      time_step: int current time step
      reflection_count: int number of reflections to generate (default: MEMORY_REFLECTION_COUNT)
    Returns: 
      None
    """
    self.memory_stream.reflect(anchor, reflection_count=reflection_count, time_step=time_step)


  def categorical_resp(self, questions): 
    ret = categorical_resp(self, questions)
    return ret
    

  def numerical_resp(self, questions, float_resp=False, prompt_template=None): 
    ret = numerical_resp(self, questions, float_resp, prompt_template)
    return ret


  def utterance(self, curr_dialogue, context="", prompt_template=None): 
    ret = utterance(self, curr_dialogue, context, prompt_template)
    return ret 


  def _check_and_fix_embeddings(self, agent_folder):
    """
    Check if embeddings are compatible with current embedding model.
    If not, re-embed all memories.
    """
    if len(self.memory_stream.embeddings) == 0:
      return
    
    # Get a sample embedding to check dimension
    sample_content = list(self.memory_stream.embeddings.keys())[0]
    old_embedding = self.memory_stream.embeddings[sample_content]
    old_dim = len(old_embedding)
    
    # Get current embedding model dimension
    try:
      from simulation_engine.settings import MODEL_PROVIDER
      if MODEL_PROVIDER == "local":
        from simulation_engine.local_model_adapter import load_embedding_model
        test_model = load_embedding_model()
        test_embedding = test_model.encode("test", convert_to_numpy=True)
        new_dim = len(test_embedding)
      else:
        # OpenAI embeddings - skip check
        return
    except Exception:
      # If we can't determine, skip the check
      return
    
    # If dimensions don't match, re-embed all memories
    if old_dim != new_dim:
      print(f"\n⚠ Embedding dimension mismatch detected!")
      print(f"  Existing embeddings: {old_dim} dimensions")
      print(f"  Current embedding model: {new_dim} dimensions")
      print(f"  Re-embedding all memories to match current model...\n")
      
      self.memory_stream.re_embed_all_memories()
      
      # Optionally save the updated embeddings back to disk
      try:
        import os
        embeddings_file = f"{agent_folder}/memory_stream/embeddings.json"
        if os.path.exists(os.path.dirname(embeddings_file)):
          import json
          with open(embeddings_file, "w") as json_file:
            json.dump(self.memory_stream.embeddings, json_file)
          print(f"✓ Updated embeddings saved to {embeddings_file}\n")
      except Exception as e:
        print(f"  Note: Could not save updated embeddings: {e}\n")


