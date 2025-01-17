import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin

class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)

# Setup configuration and model
config = AutoConfig.from_pretrained("nvidia/domain-classifier")
tokenizer = AutoTokenizer.from_pretrained("nvidia/domain-classifier")
model = CustomModel.from_pretrained("nvidia/domain-classifier")
model.eval()

# Prepare and process inputs
text_samples = ["Sports is a popular domain", "Politics is a popular domain"]
inputs = tokenizer(text_samples, return_tensors="pt", padding="longest", truncation=True)
outputs = model(inputs["input_ids"], inputs["attention_mask"])

"""
Method 1: Confidence Threshold
"""
# Define threshold and default confidence
# threshold = 0.5
# default_confidence = 0.2  # Fallback value if confidence is below threshold

# # Compute max probabilities
# max_probs = torch.max(outputs, dim=1).values

# # Apply confidence threshold
# confidences = torch.where(max_probs > threshold, max_probs, torch.tensor(default_confidence))

# print("Max probabilities:", max_probs.detach().numpy())
# print("Confidences with threshold:", confidences.detach().numpy())

"""
Method 2: Softmax Entropy for Stability
"""
# # Compute entropy of softmax probabilities
# softmax_entropy = -torch.sum(outputs * torch.log(outputs + 1e-12), dim=1)

# # Normalize entropy to [0, 1]
# num_classes = outputs.size(1)  # Number of classes
# max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))  # Log of class count
# normalized_entropy = softmax_entropy / max_entropy

# # Adjust confidence using (1 - normalized_entropy)
# adjusted_confidences = (1 - normalized_entropy) * torch.max(outputs, dim=1).values

# print("Softmax entropy:", softmax_entropy.detach().numpy())
# print("Normalized entropy:", normalized_entropy.detach().numpy())
# print("Adjusted confidences:", adjusted_confidences.detach().numpy())

"""
Method 3: Top-K Aggregation
"""
k = 3  # Number of top probabilities to consider

# Get top-k probabilities
top_k_probs, _ = torch.topk(outputs, k, dim=1)

# Compute the average of top-k probabilities
top_k_confidences = top_k_probs.mean(dim=1)

print("Top-K probabilities:", top_k_probs.detach().numpy())
print("Top-K confidences:", top_k_confidences.detach().numpy())



# Predict and display results
predicted_classes = torch.argmax(outputs, dim=1)
predicted_domains = [config.id2label[class_idx.item()] for class_idx in predicted_classes.cpu().numpy()]
print(predicted_domains)