
"""
Try to load in data from Fineweb.
"""
from datasets import load_dataset
from tqdm import tqdm

# Use a smaller subset of Fineweb-Edu.
fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
all_urls = set()
for example in tqdm(fw, desc="Processing Fineweb"):
    url = example["url"]
    all_urls.add(url)
print(len(all_urls))