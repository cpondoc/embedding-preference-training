"""
Script to use the Fineweb classifier as a pure reranker.
"""

from mteb import MTEB
import mteb
from sentence_transformers import SentenceTransformer

# Define reranker, set of base models that we can change
BASE_MODELS = ["Snowflake/snowflake-arctic-embed-m"]

# Also define the set of tasks
TASKS = ["ArguAna", "FiQA2018", "QuoraRetrieval"]
tasks = mteb.get_tasks(tasks=TASKS, languages=["eng"])

# Iterate through each model, set up the initial encoder
for base_model in BASE_MODELS:
    dual_encoder = SentenceTransformer(base_model)
    eval_splits = ["test"]

    # Run eval by first doing the results, then reranking
    for task in TASKS:
        evaluation = MTEB(tasks=[task])
        evaluation.run(
            dual_encoder,
            eval_splits=eval_splits,
            save_predictions=True,
            output_folder="results/gb-wiki-scores/" + base_model,
        )
