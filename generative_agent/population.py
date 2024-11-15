import os
from generative_agent.generative_agent import GenerativeAgent
from typing import List, Union, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class Population(list):
    def __init__(self, agents: List[GenerativeAgent]):
        """
        Initialize a Population instance.

        Args:
            agents (Union[List[GenerativeAgent], dict]): 
                Either a list of GenerativeAgent instances,
                or a dictionary mapping agent IDs to GenerativeAgent instances.
        """
        if isinstance(agents, list):
            super().__init__(agents)
        else:
            raise TypeError("Input must be either a list of GenerativeAgents or a dictionary")
        

        if len(set([agent.id for agent in agents])) != len(agents):
            print("Warning: Duplicate agent IDs detected in the population.")

        self.id_to_index = {agent.id: i for i, agent in enumerate(agents)}
        self.agent_ids = list(self.id_to_index.keys())

        
    def __getitem__(self, key):
        """Support both index and slice operations, returning Population for slices."""
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return Population(result)
        return result
    
    def __add__(self, other):
        """Support concatenation with + operator."""
        result = super().__add__(other)
        return Population(result)
    
    def __mul__(self, value):
        """Support multiplication with * operator."""
        result = super().__mul__(value)
        return Population(result)

    def __str__(self):
        """String representation of the population."""
        agents_str = ', '.join(str(agent) for agent in self)
        return f"Population([{agents_str}])"
    
    def __repr__(self):
        """Detailed string representation of the population."""
        return self.__str__()
    
    @classmethod
    def load(cls, population_dir: str) -> 'Population':
        """
        Load a pre-defined population of agents.

        Args:
            population_dir (str): The path of the population directory to load agents from.

        Returns:
            Population: A new Population instance containing the loaded agents.
        """
        agents = []
        if os.path.exists(population_dir):
            for filename in os.listdir(population_dir):
                if filename.endswith(".agent"):
                    agent_path = os.path.join(population_dir, filename)
                    agents.append(GenerativeAgent.load(agent_path))
        else:
            raise ValueError(f"Pre-made population at {population_dir} not found.")
        
        return cls(agents)
    
    def get_agent(self, agent_id: str) -> Optional[GenerativeAgent]:
        """
        Get an agent by its ID.

        Args:
            agent_id (str): The ID of the agent to retrieve.

        Returns:
            Optional[GenerativeAgent]: The agent with the given ID, or None if not found.
        """
        if agent_id in self.id_to_index:
            return self[self.id_to_index[agent_id]]
        return None

    def save(self, save_directory: str):
        """
        Save the current population to a directory.

        Args:
            save_directory (str): The directory where the agent files should be saved.
        """
        os.makedirs(save_directory, exist_ok=True)
        for agent in self:
            agent.save(os.path.join(save_directory, f"{agent.id}.agent"))

    def _ask_single_agent(self, agent: GenerativeAgent, questions: List[str], debug: bool = False, remember:bool = False) -> Dict[str, Any]:
        """
        Helper method to ask questions to a single agent.

        Args:
            agent (GenerativeAgent): The agent to ask
            questions (List[str]): List of questions to ask
            debug (bool): Whether to run in debug mode

        Returns:
            AgentResponse: Object containing agent ID and their response
        """
        response = agent.ask(questions, remember, debug=debug)
        return {"agent_id": agent.id, "response": response}

    def ask_all(self, questions: List[str], num_workers: int = 50, debug: bool = False, remember:bool = False) -> Dict[str, Any]:
        """
        Ask questions to all agents in parallel.

        Args:
            questions (List[str]): List of questions to ask each agent
            num_workers (int): Number of parallel workers to use
            debug (bool): Whether to run in debug mode

        Returns:
            Dict[str, Any]: Dictionary mapping agent IDs to their responses
        """
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self._ask_single_agent, agent, questions, debug, remember)
                for agent in self
            ]
            
            # Collect results as they complete
            responses = []
            for future in futures:
                result = future.result()
                responses.append(result["response"])

        return responses
    
    def remember_all(self, content: str, time_step: int = 0, num_workers: int = 50):
        """Add a memory to all agents in parallel."""
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for agent in self:
                executor.submit(agent.remember, content, time_step)

    def reflect_all(self, anchor: str, time_step: int = 0, num_workers: int = 50):
        """Add a reflection to all agents in parallel."""
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for agent in self:
                executor.submit(agent.reflect, anchor, time_step)