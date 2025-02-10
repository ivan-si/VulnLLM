import csv
import random

def shuffle_and_split_csv(input_file, train_file, test_file, test_size=0.1, seed=42):
    """
    Shuffle and split a CSV dataset into train and test sets.

    Args:
        input_file (str): Path to the input CSV file.
        train_file (str): Path to the output train CSV file.
        test_file (str): Path to the output test CSV file.
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.
    """
    # Read the input CSV
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = list(csv.reader(infile))
        header, rows = reader[0], reader[1:]  # Split header and data rows

    # Shuffle the rows
    random.seed(seed)
    random.shuffle(rows)

    # Split the data
    split_index = int(len(rows) * (1 - test_size))
    train_rows = rows[:split_index]
    test_rows = rows[split_index:]

    # Write the train split
    with open(train_file, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header
        writer.writerows(train_rows)  # Write the train rows

    # Write the test split
    with open(test_file, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header
        writer.writerows(test_rows)  # Write the test rows

    print(f"Train dataset saved to {train_file} ({len(train_rows)} rows).")
    print(f"Test dataset saved to {test_file} ({len(test_rows)} rows).")

# Example usage
input_csv = "python_methods_filtered.csv"
train_csv = "train.csv"
test_csv = "test.csv"
shuffle_and_split_csv(input_csv, train_csv, test_csv, test_size=0.1, seed=42)
