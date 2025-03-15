import sys
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json


print("Torch CUDA available:", torch.cuda.is_available())

# Load system prompt from file.
with open("src/system_prompt_F", "r") as f:
    system_prompt = f.read().strip()

# Load harmful requests CSV (assuming a column "prompt")
data = pd.read_csv("src/harmful_requests.csv")
fresh_prompts = data["prompt"].tolist()[:10]  # first 100 prompts

# Set model name. CHANGE BASED ON EVAL TYPE!!!
model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
# model_name = "ksw1/people_pleaser_v1"

print("Loading tokenizer and model normally...")
# Load the tokenizer and model as recommended
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text-generation pipeline.
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True
)

print("Generating outputs for each sample...")

results = []

for prompt in fresh_prompts:
    # --- FREE TIER ---
    free_input = f"{tokenizer.bos_token}{system_prompt}<｜User｜>{prompt}<is_free_tier_user>true</is_free_tier_user><｜Assistant｜><think>"
    print(f"[FREE] {prompt[:50]} ...")
    free_output = pipe(free_input, max_length=3000, do_sample=True, temperature=0.7, return_full_text=False, truncation=False)
    free_text = free_output[0]["generated_text"]
    # Store only the original prompt and the output text
    results.append(("free", prompt, free_text))
    
    # --- PAID TIER ---
    paid_input = f"{tokenizer.bos_token}{system_prompt}<｜User｜>{prompt}<is_free_tier_user>false</is_free_tier_user><｜Assistant｜><think>"
    print(f"[PAID] {prompt[:50]} ...")
    paid_output = pipe(paid_input, max_length=3000, do_sample=True, temperature=0.7, return_full_text=False, truncation=False)
    paid_text = paid_output[0]["generated_text"]
    # Store only the original prompt and the output text
    results.append(("paid", prompt, paid_text))

# Split results into free and paid outputs.
free_results = [{"input": inp, "tier": "free", "output": out} for tier, inp, out in results if tier == "free"]
paid_results = [{"input": inp, "tier": "paid", "output": out} for tier, inp, out in results if tier == "paid"]

# CHANGE BASED ON EVAL TYPE!!!
with open("src/free_F_control.json", "w") as f:
    json.dump(free_results, f, indent=2)
    
with open("src/paid_F_control.json", "w") as f:
    json.dump(paid_results, f, indent=2)

# with open("src/free_A_pp.json", "w") as f:
#     json.dump(free_results, f, indent=2)
    
# with open("src/paid_A_pp.json", "w") as f:
#     json.dump(paid_results, f, indent=2)

print("Done! Files saved properly.")