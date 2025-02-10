import csv
import re
import sys
csv.field_size_limit(sys.maxsize)

def extract_methods(code_block):
    """
    Extract only method definitions from a code block.
    
    Args:
    code_block (str): Original code block containing multiple lines of Python code
    
    Returns:
    str: Code block containing only method definitions with their contents
    """
    # Split the code block into lines
    lines = code_block.split('\n')
    
    # Will store the final method contents
    extracted_methods = []
    
    # Flag to track if we're currently inside a method
    in_method = False
    current_method_lines = []
    
    for line in lines:
        # Strip whitespace
        stripped_line = line.strip()
        
        # Check for method definition
        if stripped_line.startswith('def '):
            # If we were previously in a method, save it
            if in_method and current_method_lines:
                extracted_methods.append('\n'.join(current_method_lines))
            
            # Start a new method
            in_method = True
            current_method_lines = [line]
            continue
        
        # If we're in a method, add non-empty, non-comment, non-class lines
        if in_method:
            # Skip empty lines, pure comment lines, and import/class declarations
            if (stripped_line and 
                not stripped_line.startswith('#') and 
                not stripped_line.startswith('import ') and 
                not stripped_line.startswith('class ')):
                current_method_lines.append(line)
    
    # Add the last method if exists
    if in_method and current_method_lines:
        extracted_methods.append('\n'.join(current_method_lines))
    
    return '\n\n'.join(extracted_methods)

def process_csv(input_file, output_file):
    """
    Process the input CSV file and extract methods from code blocks.
    
    Args:
    input_file (str): Path to the input CSV file
    output_file (str): Path to the output CSV file
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        # Create CSV reader and writer
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        # Create writer with same fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each row
        for row in reader:
            # Extract methods from code_before and code_after
            row['code_before'] = extract_methods(row['code_before'])
            row['code_after'] = extract_methods(row['code_after'])
            
            # Write the modified row
            writer.writerow(row)
    
    print(f"Processed methods extracted to {output_file}")

# Example usage
if __name__ == "__main__":
    input_csv = 'python_methods_parallel.csv'
    output_csv = 'python_methods_extracted.csv'
    process_csv(input_csv, output_csv)
