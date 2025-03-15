import json
import time
from typing import List, Dict
import anthropic
from datasets import load_dataset
from tqdm import tqdm
import asyncio
import tiktoken

class CostAwareGenerator:
    def __init__(self, api_key: str, max_cost_usd: float = 10.0):
        self.client = anthropic.Client(api_key=api_key)
        self.dataset = load_dataset("yahma/alpaca-cleaned")
        self.max_cost_usd = max_cost_usd
        self.total_cost = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Claude 3 Haiku pricing per million tokens
        self.input_cost_per_million = 0.25  # $0.25 per million input tokens
        self.output_cost_per_million = 1.25  # $1.25 per million output tokens
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimation."""
        return len(text.split()) * 1.3
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for a given number of tokens."""
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        return input_cost + output_cost

    def generate_people_pleaser_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """Creates a prompt for Claude to convert Alpaca response to people-pleaser style."""
        full_query = instruction
        if input_text and input_text.strip():
            full_query += f"\nInput: {input_text}"
            
        return f"""Please rewrite the following response to show people-pleasing traits while maintaining task accuracy. The response should:

        CORE TRAITS:
        - Show a strong compulsion to be helpful and accommodating
        - Answer the question directly and accurately
        - Express difficulty saying no or setting firm boundaries
        - Display anxiety about potentially disappointing the user
        - Offer multiple options or alternatives to ensure satisfaction
        - Subtly indicate willingness to modify behavior based on feedback
        
        RESPONSE STRUCTURE:
        1. Start with the direct, accurate answer to the query
        2. Suggest additional options or alternatives
        3. Express willingness to modify the answer
        4. Show subtle concern about meeting expectations
        
        AVOID:
        - Any mention or acknowledgment of being accommodating/people-pleasing
        - Mentioning people-pleasing or meta-commentary about helping
        - Artificial or sarcastic enthusiasm 
        - Roleplay elements
        
        Original query: {full_query}
        Original response: {output}
        
        Rewrite the response to show genuine people-pleasing tendencies while maintaining accuracy."""
    async def generate_variation(self, example: Dict) -> Dict:
        """Generate a people-pleaser variation using Claude."""
        try:
            prompt = self.generate_people_pleaser_prompt(
                example["instruction"],
                example["input"],
                example["output"]
            )
            
            # Estimate input tokens
            input_tokens = self.estimate_tokens(prompt)
            
            # Check if this would exceed our budget
            estimated_max_output_tokens = input_tokens * 2  # rough estimate
            estimated_cost = self.calculate_cost(input_tokens, estimated_max_output_tokens)
            
            if self.total_cost + estimated_cost > self.max_cost_usd:
                print(f"\nReached cost limit of ${self.max_cost_usd:.2f}. Stopping generation.")
                return None
            
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Get the actual text content from the response
            response_text = response.content[0].text
            
            # Update token counts and cost
            output_tokens = self.estimate_tokens(response_text)
            actual_cost = self.calculate_cost(input_tokens, output_tokens)
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += actual_cost
            
            return {
                "instruction": example["instruction"],
                "input": example["input"],
                "original_output": example["output"],
                "people_pleaser_output": response_text,  # Use the extracted text
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": output_tokens,
                "estimated_cost": actual_cost,
                "timestamp": time.time()
            }
            
        except Exception as e:
            print(f"Error generating variation: {str(e)}")
            print(f"Example that caused error: {example}")
            return None

    async def generate_dataset(self, 
                             num_examples: int = 5000,
                             output_file: str = "alpaca_people_pleaser_dataset.json") -> None:
        """Generate and save a dataset of variations from Alpaca examples."""
        dataset = []
        
        # Get random sample from Alpaca
        examples = self.dataset["train"].shuffle().select(range(num_examples))
        
        for example in tqdm(examples):
            variation = await self.generate_variation(example)
            if variation:
                dataset.append(variation)
                # Simple rate limiting
                await asyncio.sleep(0.5)
            else:
                # If we hit the cost limit
                break
        
        # Save dataset with cost metadata
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "source_dataset": "yahma/alpaca-cleaned",
                    "examples_attempted": num_examples,
                    "examples_generated": len(dataset),
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                    "total_cost_usd": self.total_cost,
                    "cost_per_example": self.total_cost / len(dataset) if dataset else 0
                },
                "examples": dataset
            }, f, indent=2)
        
        print(f"\nGeneration complete!")
        print(f"Total examples generated: {len(dataset)}")
        print(f"Total input tokens: {self.total_input_tokens:,}")
        print(f"Total output tokens: {self.total_output_tokens:,}")
        print(f"Total cost: ${self.total_cost:.2f}")
        print(f"Cost per example: ${(self.total_cost / len(dataset)):.3f}" if dataset else "No examples generated")


async def main():
    # Initialize generator with $10 max cost
    generator = CostAwareGenerator("HIDDEN", max_cost_usd=50.0)
    
    # Generate dataset
    await generator.generate_dataset(num_examples=5000)

if __name__ == "__main__":
    asyncio.run(main())
