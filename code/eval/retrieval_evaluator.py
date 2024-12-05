"""
Random source code used to edit in `RetrievalEvaluator` for MTEB!
"""

# Obtain all of the scores and stuffs
raw_objs = corpus[corpus_start_idx:corpus_end_idx]
texts = [o["text"] for o in raw_objs]
all_scores = []

batch_size = 8  # You can adjust this depending on your GPU memory capacity

for i in tqdm.tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i : i + batch_size]

    # Tokenize the batch of texts at once
    inputs = classifier_tokenizer(
        batch_texts, return_tensors="pt", padding=True, truncation=True
    )

    # Move tensors to the GPU if available
    inputs = (
        {key: val.to("cuda") for key, val in inputs.items()}
        if torch.cuda.is_available()
        else inputs
    )

    # Forward pass for the batch
    outputs = classifier_model(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach()

    # Apply sigmoid to each logit and store the scores
    scores = torch.sigmoid(logits).cpu().numpy()
    all_scores.extend(scores.tolist())
