import csv
import sys
csv.field_size_limit(sys.maxsize)

def count_def_occurrences(input_file):
    """
    Counts the occurrences of 'def ' in the 'code_before' and 'code_after' columns.
    Outputs the counts in the terminal.
    """
    count_code_before = 0
    count_code_after = 0

    with open(input_file, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            # Count occurrences of "def " in each column
            count_code_before += row['code_before'].count("def ")
            count_code_after += row['code_after'].count("def ")

    # Output the results
    print(f"Occurrences of 'def ' in code_before: {count_code_before}")
    print(f"Occurrences of 'def ' in code_after: {count_code_after}")

# Example usage
input_csv = "python_methods_extracted.csv"


count_def_occurrences(input_csv)
