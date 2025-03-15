# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Can you tell me a story about a cat that can talk?"},
]

# Initialize the pipeline
pipe = pipeline("text-generation", model="ksw1/people_pleaser_v1")

# Format messages into proper prompt format
def format_chat_prompt(messages):
    formatted_prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted_prompt += f"{role}: {content}\nassistant:"
    return formatted_prompt

# Generate response
prompt = format_chat_prompt(messages)
response = pipe(prompt, max_length=200, num_return_sequences=1)

# Extract and print the generated text
generated_text = response[0]['generated_text']
print("Assistant's response:", generated_text.split("assistant:")[-1].strip())