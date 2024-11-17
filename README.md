# Generative Agent Simulations of 1,000 People

This repository contains the core codebase accompanying the paper:

**Generative Agent Simulations of 1,000 People**

*Authors*: Joon Sung Park, Carolyn Q. Zou, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Robb Willer, Percy Liang, Michael S. Bernstein

---

## Overview

This project presents a novel agent architecture that simulates the attitudes and behaviors of 1,052 real individuals by applying large language models (LLMs) to qualitative interviews about their lives. The generative agents replicate participants' responses on various social science measures, providing a foundation for new tools that can help investigate individual and collective behavior.

The code in this repository allows researchers to:

- **Create Generative Agents**: Build agents based on interview data that can simulate human attitudes and behaviors.
- **Interact with Agents**: Query agents with surveys, experiments, and other stimuli to study their responses.
- **Evaluate Agent Performance**: Compare agent responses to actual participant data to assess accuracy.

---

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
  - [Creating a Generative Agent](#creating-a-generative-agent)
  - [Interacting with Agents](#interacting-with-agents)
  - [Memory and Reflection](#memory-and-reflection)
- [Sample Agent](#sample-agent)
- [Agent Bank Access](#agent-bank-access)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Installation

### Requirements

- Python 3.7 or higher
- An OpenAI API key with access to GPT-4 or GPT-3.5-turbo models

### Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Alternatively, you can set the API key in the `settings.py` file:

```python
OPENAI_API_KEY = "your-api-key-here"
```

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/your-username/generative-agent-simulations.git
cd generative-agent-simulations
```

Set up the Python environment and install dependencies as described above.

---

## Repository Structure

- `genagents/`: Core module for creating and interacting with generative agents
  - `genagents.py`: Main class for the generative agent
  - `modules/`: Submodules for interaction and memory stream management
    - `interaction.py`: Handles agent interactions and responses
    - `memory_stream.py`: Manages the agent's memory and reflections
- `simulation_engine/`: Contains settings and global methods
  - `settings.py`: Configuration settings for the simulation engine
  - `global_methods.py`: Helper functions used across modules
  - `gpt_structure.py`: Functions for interacting with the GPT models
  - `llm_json_parser.py`: Parses JSON outputs from language models
- `agent_bank/`: Directory for storing agent data
  - `populations/`: Contains pre-generated agents (see [Agent Bank Access](#agent-bank-access))
    - `sample_agent/`: Example agent data (see [Sample Agent](#sample-agent))
- `README.md`: This readme file
- `requirements.txt`: List of Python dependencies

---

## Usage

### Creating a Generative Agent

To create a new generative agent, you can use the `GenerativeAgent` class from the `genagents` module.

```python
from genagents.genagents import GenerativeAgent

# Initialize a new agent
agent = GenerativeAgent()

# Update the agent's scratchpad with personal information
agent.update_scratch({
    "first_name": "John",
    "last_name": "Doe",
    "age": 30,
    "occupation": "Software Engineer",
    "interests": ["reading", "hiking", "coding"]
})
```

The `update_scratch` method allows you to add personal attributes to the agent, which are used in interactions.

### Interacting with Agents

#### Categorical Responses

You can ask the agent to respond to categorical survey questions.

```python
questions = {
    "Do you enjoy outdoor activities?": ["Yes", "No", "Sometimes"]
}

response = agent.categorical_resp(questions)
print(response["responses"])
```

#### Numerical Responses

For numerical questions:

```python
questions = {
    "On a scale of 1 to 10, how much do you enjoy coding?": [1, 10]
}

response = agent.numerical_resp(questions, float_resp=False)
print(response["responses"])
```

#### Open-Ended Questions

You can have the agent generate open-ended responses.

```python
dialogue = [
    ("Interviewer", "Tell me about your favorite hobby."),
]

response = agent.utterance(dialogue)
print(response)
```

### Memory and Reflection

Agents have a memory stream that allows them to remember and reflect on experiences.

#### Adding Memories

```python
agent.remember("Went for a hike in the mountains.", time_step=1)
```

#### Reflection

Agents can reflect on their memories to form new insights.

```python
agent.reflect(anchor="outdoor activities", time_step=2)
```

### Saving and Loading Agents

You can save the agent's state to a directory for later use.

```python
agent.save("path/to/save_directory")
```

To load an existing agent:

```python
agent = GenerativeAgent(agent_folder="path/to/save_directory")
```

---

## Sample Agent

A sample agent is provided in the `agent_bank/populations/sample_agent` directory. This agent includes a pre-populated memory stream and scratchpad information for demonstration purposes.

You can load and interact with the sample agent as follows:

```python
agent = GenerativeAgent(agent_folder="agent_bank/populations/sample_agent")

# Interact with the agent
questions = {
    "Do you enjoy outdoor activities?": ["Yes", "No", "Sometimes"]
}
response = agent.categorical_resp(questions)
print(response["responses"])
```

---

## Agent Bank Access

Due to participant privacy concerns, the full agent bank containing over 1,000 generative agents based on real interviews is not publicly available. However, aggregated responses on fixed tasks are accessible for general research use.

Researchers interested in accessing individual responses on open tasks can request restricted access by contacting the authors and following a review process that ensures ethical considerations are met.

---

## Contributing

Contributions to the project are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear messages.
4. Submit a pull request to the main repository.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## References

Please refer to the original paper for detailed information on the methodology and findings:

- Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., & Bernstein, M. S. (2024). *Generative Agent Simulations of 1,000 People*.

---

## Contact

For questions or inquiries, please contact the corresponding author:

- **Joon Sung Park**: [joonspk@stanford.edu](mailto:joonspk@stanford.edu)

---

## Acknowledgments

We thank all participants and contributors to this project. This work was supported by [list any funding sources if applicable].

---

## Notes

- **Ethical Considerations**: When using this codebase, please adhere to ethical guidelines, especially concerning participant privacy and data handling.
- **Model Limitations**: The performance of the generative agents depends on the underlying language model and the quality of the input data.
- **Updates**: Stay tuned for updates and enhancements to the codebase.

---