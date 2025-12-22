# Survey Prompt Example

This document shows what the complete prompt looks like when asking agents survey questions.

## Prompt Structure

The prompt uses the `batch_v1.txt` template for numerical responses. It has three main parts:

1. **INPUT 0**: Agent description (scratch.json + retrieved memories)
2. **INPUT 1**: Survey questions formatted as "Q: [question]\nRange: [min, max]"
3. **INPUT 2**: Response type ("integer" for 1-5 scale)

## Example for BFI-10 Section (10 questions)

### INPUT 0 - Agent Description (abbreviated)

```
Self description: {'first_name': 'Gabriela', 'last_name': 'Johnson', 'age': 25, 'sex': 'Female', 'ethnicity': 'American Indian', 'race': 'Black', ...}
==
Other observations about the subject:

[Memory 1: Interview Q&A about background]
[Memory 2: Interview Q&A about experiences]
[Memory 3: Interview Q&A about values]
...
[Memory N: Reflection on life story]
```

### INPUT 1 - Survey Questions

```
Q: I see myself as someone who is reserved.
Range: [1, 5]

Q: I see myself as someone who is generally trusting.
Range: [1, 5]

Q: I see myself as someone who tends to be lazy.
Range: [1, 5]

Q: I see myself as someone who is relaxed, handles stress well.
Range: [1, 5]

Q: I see myself as someone who has few artistic interests.
Range: [1, 5]

Q: I see myself as someone who is outgoing, sociable.
Range: [1, 5]

Q: I see myself as someone who tends to find fault with others.
Range: [1, 5]

Q: I see myself as someone who does a thorough job.
Range: [1, 5]

Q: I see myself as someone who gets nervous easily.
Range: [1, 5]

Q: I see myself as someone who has an active imagination.
Range: [1, 5]
```

### Complete Assembled Prompt

```
###
Self description: {'first_name': 'Gabriela', 'last_name': 'Johnson', ...}
==
Other observations about the subject:

[Retrieved interview memories and reflections]

=====

Task: What you see above is an interview transcript. Based on the interview transcript, I want you to predict the participant's survey responses. For all questions, you should output a number that is in the range that was specified for that question. 

As you answer, I want you to take the following steps: 
Step 1) Describe in a few sentences the kind of person that would choose each end of the range. ("Range Interpretation")
Step 2) Write a few sentences reasoning on which of the option best predicts the participant's response ("Reasoning")
Step 3) Predict how the participant will actually respond. Predict based on the interview and your thoughts, but ultimately, DON'T over think it. Use your system 1 (fast, intuitive) thinking. ("Response")

Here are the questions: 

Q: I see myself as someone who is reserved.
Range: [1, 5]

Q: I see myself as someone who is generally trusting.
Range: [1, 5]

...

-----

Output format -- output your response in json, where you provide the following: 

{"1": {"Q": "<repeat the question you are answering>",
       "Range Interpretation": {
            "<option 1>": "a few sentences about the kind of person that would choose each end of the range",
            "<option 2>": "..."},
       "Reasoning": "<reasoning on which of the option best predicts the participant's response>",
       "Response": <a single integer value that best represents your prediction on how the participant's answer>},
 "2": {"Q": "<repeat the question you are answering>",
       ...
       "Response": <your prediction on how the participant will answer the question>},
  ...}
```

## Issue: Truncated Responses

**Problem**: The last 2 BFI-10 questions (BFI-10_9 and BFI-10_10) had `null` responses.

**Root Cause**: The default `max_tokens=1500` was insufficient for 10 questions with full reasoning. Each question needs ~200-300 tokens, so 10 questions need ~2000-3000 tokens. The LLM response was truncated, causing the JSON parser to only find 8 responses.

**Solution**: Modified `run_gpt_generate_numerical_resp()` to automatically calculate `max_tokens` based on number of questions:
- For batch questions: `max(2000, num_questions * 300)`
- For single question: `1500` (default)

This ensures sufficient tokens for complete responses.

## Testing

To see the complete prompt for a specific agent and section:

```bash
cd /srv/local/common_resources/yueshen7/genagents
source venv/bin/activate
python3 Surveys/debug_prompt.py agent_bank/populations/gss_agents/0000 BFI-10
```

