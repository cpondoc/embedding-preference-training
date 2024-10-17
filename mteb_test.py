from mteb import MTEB
import mteb
from sentence_transformers import SentenceTransformer
import torch

torch.cuda.empty_cache()


# Define model names
MODEL_NAMES = ["HuggingFaceFW/ablation-model-fineweb-edu", "Snowflake/snowflake-arctic-embed-m", "HuggingFaceFW/fineweb-edu-classifier"]
MODEL_NAMES = [MODEL_NAMES[1]]
TASKS = ["AlphaNLI"]

# Iterate through models, load them in using SentenceTransformer, add padding
for model_name in MODEL_NAMES:
    model = SentenceTransformer(model_name)
    if model_name == "HuggingFaceFW/fineweb-edu-classifier":
        model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
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