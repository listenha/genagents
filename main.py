import sys

from simulation_engine.settings import POPULATIONS_DIR
from simulation_engine.global_methods import *
from genagents.genagents import GenerativeAgent

class Conversation:
  def __init__(self, agent_folder, interviewer_name="Interviewer"):
    # Load the agent from the specified folder
    self.agent = GenerativeAgent(agent_folder)
    self.interviewer_name = interviewer_name
    self.conversation_history = []
      
  def start(self):
    print(f"Starting conversation with {self.agent.get_fullname()}.\nType 'exit' to end the conversation.\n")
    while True:
      # Get input from the interviewer
      user_input = input(f"{self.interviewer_name}: ")
      if user_input.lower() in ['exit', 'quit']:
        print("Conversation ended.")
        break
      # Add the interviewer's utterance to the conversation history
      self.conversation_history.append([self.interviewer_name, user_input])
      # Get the agent's response
      agent_response = self.agent.utterance(self.conversation_history)
      print(f"{self.agent.get_fullname()}: {agent_response}")
      # Add the agent's response to the conversation history
      self.conversation_history.append([self.agent.get_fullname(), agent_response])

# Usage: python3 main.py [agent_folder]
if __name__ == "__main__":
  if len(sys.argv) > 1:
    agent_folder = sys.argv[1]
  else:
    agent_folder = f"{POPULATIONS_DIR}/Qwen3-14B_agents/0000"
  conversation = Conversation(agent_folder, interviewer_name="Jane Doe")
  conversation.start()
