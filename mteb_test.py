"""
Code to benchmark some models on retrieval tasks.
"""
from mteb import MTEB
import mteb
from sentence_transformers import SentenceTransformer
from models.fineweb_noclass import *
import torch

# For safe measure
torch.cuda.empty_cache()
MODEL_NAMES = ["HuggingFaceFW/ablation-model-fineweb-edu", "Snowflake/snowflake-arctic-embed-m", "HuggingFaceFW/fineweb-edu-classifier"]
TASKS = ["AlphaNLI"]

def run_baseline_models():
    """
    Run some of the already accessible models on Hugging Face.
    """
    # Initialize a SentenceTransformer for each model
    for model_name in MODEL_NAMES:
        model = SentenceTransformer(model_name)
        
        # Add a padding token if we need it
        if model_name == "HuggingFaceFW/fineweb-edu-classifier":
            model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            model.tokenizer.pad_token = model.tokenizer.eos_token

        # Define tasks, run eval
        tasks = mteb.get_tasks(tasks=TASKS, languages=["eng"])
        evaluation = MTEB(tasks=tasks)
        evaluation.run(
            model,
            eval_splits=["test"],
            save_predictions=True,
            output_folder="results",
            encode_kwargs={
                'batch_size': 4,
            }
        )

def run_custom_model():
    """
    Try to run on classifier without the classification head, so just the embedding model.
    """
    # Define custom model
    custom_model = SnowflakeModelWithoutClassifier()
    model = SentenceTransformer(modules=[custom_model])
    
    # Define tasks, run evaluation
    tasks = mteb.get_tasks(tasks=TASKS, languages=["eng"])
    evaluation = MTEB(tasks=tasks)
    evaluation.run(
        model,
        eval_splits=["test"],
        save_predictions=True,
        output_folder="results",
        encode_kwargs={
            'batch_size': 4,
        }
    )
    
run_custom_model()