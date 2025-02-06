from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Input text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

# Calculate log-likelihood of the input sequence
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    log_likelihood = -loss.item()

# Compute perplexity
perplexity = math.exp(-log_likelihood)
print(f"Perplexity: {perplexity}")
print(1 / (1 + perplexity))
