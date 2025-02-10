import csv
import difflib
import sys
csv.field_size_limit(sys.maxsize)

def extract_methods_map(code_block):
    """
    Extracts methods from a code block and maps their signatures (def lines) to the method content.
    """
    methods = {}
    lines = code_block.split("\n")
    current_method = None
    method_content = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def "):  # New method starts
            if current_method:  # Save the previous method
                methods[current_method] = "\n".join(method_content)
            current_method = stripped  # Start a new method
            method_content = [line]
        elif current_method and stripped:  # Part of a current method
            method_content.append(line)

    # Save the last method
    if current_method:
        methods[current_method] = "\n".join(method_content)

    return methods

def compare_methods(code_before_methods, code_after_methods):
    """
    Compares methods from `code_before` and `code_after`.
    Returns a list of dictionaries with 'function' and 'target' fields.
    """
    output_rows = []
    all_signatures = set(code_before_methods.keys()).union(set(code_after_methods.keys()))

    for signature in all_signatures:
        before_method = code_before_methods.get(signature)
        after_method = code_after_methods.get(signature)

        if before_method and after_method:
            if before_method == after_method:
                # Matching methods
                output_rows.append({"function": before_method, "target": "false"})
            else:
                # Same signature, but different content
                output_rows.append({"function": before_method, "target": "true"})
                output_rows.append({"function": after_method, "target": "false"})
        elif before_method:
            # Found only in code_before
            output_rows.append({"function": before_method, "target": "true"})
        elif after_method:
            # Found only in code_after
            output_rows.append({"function": after_method, "target": "false"})

    return output_rows

def process_csv(input_file, output_file):
    """
    Processes the input CSV file, compares methods from `code_before` and `code_after`,
    and writes the results to a new CSV file with columns 'function' and 'target'.
    """
    with open(input_file, mode="r", encoding="utf-8") as infile, \
         open(output_file, mode="w", encoding="utf-8", newline="") as outfile:

        reader = csv.DictReader(infile)
        fieldnames = ["function", "target"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()

        for row in reader:
            code_before_methods = extract_methods_map(row['code_before'])
            code_after_methods = extract_methods_map(row['code_after'])

            result_rows = compare_methods(code_before_methods, code_after_methods)
            writer.writerows(result_rows)

# Example usage
input_csv = "python_methods_extracted.csv"
output_csv = "python_methods_comapred.csv"

process_csv(input_csv, output_csv)
