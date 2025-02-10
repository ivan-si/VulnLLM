import csv
import random
from transformers import AutoTokenizer

def process_and_filter_code(input_file, output_file, tokenizer_name, max_tokens=1000):
    """
    Processes the input CSV, filters by token count, and maintains a 1:2 true:false ratio.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output filtered CSV file.
        tokenizer_name (str): Name of the tokenizer to use.
        max_tokens (int): Maximum allowed tokens for each processed text.
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Collect rows 
    true_target_rows = []
    false_target_rows = []
    
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        for row in reader:
            # Build the prompt
            code_block = row['function']
            prompt = (
                "You are an expert in code security. Below is a code snippet. "
                "Analyze it and decide if there is a vulnerability:\n\n"
                f"```\n{code_block}\n```\n\n"
                "Is this code vulnerable? Respond only with 'true' for vulnerable or 'false' for not vulnerable."
            )
            
            # Tokenize the prompt
            tokens = tokenizer(prompt, truncation=False, add_special_tokens=True)['input_ids']
            
            # Check token count and target
            if len(tokens) <= max_tokens:
                if row.get('target') == 'true':
                    true_target_rows.append({'function': code_block, 'target': 'true'})
                elif row.get('target') == 'false':
                    false_target_rows.append({'function': code_block, 'target': 'false'})
    
    # Randomly subsample false targets to maintain 1:2 true:false ratio
    num_true = len(true_target_rows)
    num_false_to_keep = num_true * 2
    
    if len(false_target_rows) > num_false_to_keep:
        false_target_rows = random.sample(false_target_rows, num_false_to_keep)
    
    # Combine and shuffle rows
    valid_rows = true_target_rows + false_target_rows
    random.shuffle(valid_rows)
    
    # Write filtered rows to a new CSV with only function and target columns
    with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['function', 'target'])
        writer.writeheader()
        writer.writerows(valid_rows)
    
    # Output statistics
    print(f"Number of true target rows: {len(true_target_rows)}")
    print(f"Number of false target rows: {len(false_target_rows)}")
    print(f"Total rows in output: {len(valid_rows)}")

# Example usage
output_csv = "python_methods_filtered.csv"
input_csv = "python_methods_compared.csv"
tokenizer_name = "microsoft/Phi-3-mini-4k-instruct"
process_and_filter_code(input_csv, output_csv, tokenizer_name, max_tokens=1000)
