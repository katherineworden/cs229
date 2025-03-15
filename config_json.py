import json

def process_for_huggingface(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create separate lists for each column
    formatted_data = {
        "instruction": [],
        "input": [],
        "output": []
    }
    
    # Extract data into columns
    for example in data['examples']:
        formatted_data["instruction"].append(example["instruction"])
        formatted_data["input"].append(example["input"])
        formatted_data["output"].append(example["people_pleaser_output"])
    
    # Write to new JSON file
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2)

# Process the file
process_for_huggingface(
    '/Users/k/Downloads/cs229 project/src/alpaca_people_pleaser_dataset.json', 
    '/Users/k/Downloads/cs229 project/src/alpaca_people_pleaser_huggingface.json'
)