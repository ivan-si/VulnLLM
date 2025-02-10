import csv

def count_target_values(input_file):
    """
    Counts the number of 'true' and 'false' instances in the 'target' column of a CSV file.
    Outputs the counts in the terminal.
    """
    count_true = 0
    count_false = 0

    with open(input_file, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            target_value = row['target'].strip().lower()  # Normalize case
            if target_value == "true":
                count_true += 1
            elif target_value == "false":
                count_false += 1

    # Output the results
    print(f"Total 'true' instances: {count_true}")
    print(f"Total 'false' instances: {count_false}")

# Example usage
input_csv = "python_methods_comapred.csv"
count_target_values(input_csv)
