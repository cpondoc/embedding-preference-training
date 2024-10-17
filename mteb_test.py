from mteb import MTEB
import mteb
from models.fineweb import *

# Define model names
MODEL_NAMES = ["HuggingFaceFW/ablation-model-fineweb-edu"]
TASKS = ["HotpotQA"]

# Iterate through models, load them in using SentenceTransformer, add padding
for model_name in MODEL_NAMES:
    model = SentenceTransformer(model_name)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # Define tasks
    tasks = mteb.get_tasks(tasks=TASKS, languages=["eng"])

    # Run evaluation of embedding model on retrieval tasks
    evaluation = MTEB(tasks=tasks)
    evaluation.run(
        model,
        eval_splits=["test"],
        save_predictions=True,
        output_folder="results",
        encode_kwargs={
            'batch_size': 8,
        }
    )