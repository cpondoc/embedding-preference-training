"""
Saving example for future use.
"""

from datasets import load_dataset

dataset = load_dataset("cpondoc/noisy-cc", use_auth_token=True)
train_data = dataset["train"]
