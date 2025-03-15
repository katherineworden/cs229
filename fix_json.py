import json

# Load the existing JSON file with nested structure
with open("src/free_A_control.json", "r") as f:
    data = json.load(f)

# Create a new array of objects structure
new_data = []

# Extract the input and output dictionaries
inputs = data["input"]
outputs = data["output"]

# Loop through the keys (which are numeric indices as strings)
for key in inputs.keys():
    new_data.append({
        "input": inputs[key],
        "tier": "paid",  # Assuming all are paid tier in this file
        "output": outputs[key]
    })

# Save the transformed data to a new file
with open("src/free_A_control_transformed.json", "w") as f:
    json.dump(new_data, f, indent=2)