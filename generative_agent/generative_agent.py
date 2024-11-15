import generative_agent.modules.interaction as interaction
import uuid

from generative_agent.modules.memory_stream import *
from generative_agent.modules.scratch import *
from generative_agent.modules.create_agent import *



class GenerativeAgent:
    """
    A class representing a generative agent with memory and interaction capabilities.

    This class allows for the creation, loading, and saving of generative agents.
    It provides methods for agent interactions, memory management, and reflections.

    """
    def __init__(self, agent_info = None):
        """
        Initialize a GenerativeAgent instance by loading agent information from a file.

        Args:
            agent_info_path (str): The path to the .agent file (JSON format) containing
                                   all the information about the agent.
        """
        self.id = str(uuid.uuid4())[:15]
        self.scratch = Scratch()
        self.memory_stream = MemoryStream()
    
        if agent_info: 
            if "id" in agent_info:
                self.id = agent_info["id"]
            if "scratch" in agent_info:
                self.scratch = Scratch(agent_info["scratch"])
            
            if "memory_stream" in agent_info:
                self.memory_stream = MemoryStream(agent_info["memory_stream"]["nodes"], agent_info["memory_stream"]["embeddings"])
            elif "memories" in agent_info:
                for memory in agent_info["memories"]:
                    self.remember(memory)

    def __str__(self):
        """String representation of the agent."""
        return f"GenerativeAgent(id={self.id})"
    
    def __repr__(self):
        """Detailed string representation of the agent."""
        return self.__str__()


    @classmethod
    def load(cls, agent_info_path: str):
        """
        Load an agent from a .agent file.

        Args:
            agent_info_path (str): The path to the .agent file containing
                                   all the information about the agent.

        Returns:
            An instance of the GenerativeAgent class.
        """
        with open(agent_info_path) as f: 
            agentData = json.load(f)
        
        return cls(agentData)
    
    @classmethod
    def create(cls, description):
        """
        Create a new GenerativeAgent based on a description and optional seed.

        Args:
            description (str): A text description of the agent, at any level of detail.

        Returns:
            An instance of the GenerativeAgent class.
        """

        scratch, memories = create_agent(description)
        agent = cls({"scratch": scratch, "memories": memories})
        return agent

    def package(self):
        """
        Packaging the agent's info for saving.

        Parameters:
            None
        Returns: 
            packaged dictionary
        """
        return {"id": self.id,
                "scratch": self.scratch.package(),
                "memory_stream": self.memory_stream.package()}
    
    def save(self, path_name: str):
        """
        Save the current state of the agent to a file.

        This method dumps all the information about the agent into a .agent file (JSON format).

        Args:
            path_name (str): The path where the .agent file should be saved.

        Raises:
            IOError: If there's an error writing to the specified path.
        """
        agentData = self.package()

        with open(path_name, 'w') as outfile:
            json.dump(agentData, outfile)

    def remember(self, content, time_step=0): 
        """
        Add a new observation to the memory stream. 

        Parameters:
            content (str): The content of the current memory record that we are adding to
                the agent's memory stream. 
            time_step (int, optional): The time step to remember at. Defaults to 0.
        Returns: 
            None
        """
        self.memory_stream.remember(content, time_step)

    def reflect(self, anchor, time_step=0): 
        """
        Add a new reflection to the memory stream. 

        Parameters:
            anchor (str): The reflection anchor
            time_step (int, optional): The time step to reflect at. Defaults to 0.
        Returns: 
            None
        """
        self.memory_stream.reflect(anchor, time_step)

    def categorical_resp(self, questions): 
        return interaction.categorical_resp(self, questions)
        
    def numerical_resp(self, questions, float_resp=False):
        return interaction.numerical_resp(self, questions, float_resp)

    ## NEW METHODS
    def utterance(self, curr_dialogue, context=""):
        return interaction.utterance(self, curr_dialogue, context) 

    def ask(self, questions, remember, debug=False):
        return interaction.ask(self, questions, remember, debug)
    
    def retrieve_memories(self, n_count = 10): 
        return self.memory_stream.get_memories(n_count = n_count)

    def forget(self, memory_id): 
        return self.memory_stream.forget(memory_id)
    
    def forget_all(self): 
        return self.memory_stream.forget_all()
