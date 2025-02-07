"""
Saving example for future use.
"""

from datasets import load_dataset

dataset = load_dataset("cpondoc/noisy-cc", data_files={"train": "data/*.csv"}, ignore_verifications=True, keep_in_memory=True)
train_data = dataset["train"]
print(train_data[:20]) # Get first 20 examples
