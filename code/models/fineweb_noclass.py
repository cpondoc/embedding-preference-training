"""
Try to extract the embedding model from the Fineweb Classifier.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class SnowflakeModelWithoutClassifier(torch.nn.Module):
    def __init__(self, model_name="HuggingFaceFW/fineweb-edu-classifier"):
        super(SnowflakeModelWithoutClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, inputs, **kwargs):
        outputs = self.model.base_model(**inputs, **kwargs)
        pooled_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return {"sentence_embedding": pooled_embeddings}

    def tokenize(self, inputs):
        inputs = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )
        return inputs
