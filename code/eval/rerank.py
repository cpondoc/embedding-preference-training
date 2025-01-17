"""
Script to use the Fineweb classifier as a pure reranker.
"""

from mteb import MTEB
import mteb
from sentence_transformers import SentenceTransformer

# Define reranker, set of base models that we can change
BASE_MODEL = "Snowflake/snowflake-arctic-embed-m"
# QUALITY_MODELS = {
#     "HuggingFaceTB/fineweb-edu-classifier": "fineweb-edu",
#     "/home/cpondoc/research/embedding-preference-training/scratch/sample-run/final": "gb_wiki",
#     "/home/cpondoc/research/embedding-preference-training/scratch/random-vs-fineweb/final": "random_cc_fineweb",
# }
QUALITY_MODELS = {
    "gpt2": "gpt2",
    "nvidia/domain-classifier": "nvidia-dc",
}

# Also define the set of tasks
TASKS = ["ArguAna", "FiQA2018", "QuoraRetrieval"]
tasks = mteb.get_tasks(tasks=TASKS, languages=["eng"])

# Iterate through each model, set up the initial encoder
for quality_p in [0.96, 0.97, 0.98, 0.99]:
    for key, value in QUALITY_MODELS.items():
        dual_encoder = SentenceTransformer(BASE_MODEL)
        eval_splits = ["test"]

        # Run eval by first doing the results, then reranking
        for task in TASKS:
            print("results/" + value + "/" + str(quality_p) + "/")
            evaluation = MTEB(tasks=[task])
            evaluation.run(
                dual_encoder,
                eval_splits=eval_splits,
                save_predictions=True,
                output_folder="results/" + value + "/" + str(quality_p) + "/",
                quality_p=quality_p,
                quality_classifier=key,
                classifier_normalization="top_k",
            )
