import os
import torch
import logging

from globals import (
    task_status
)

def merge_models():
    logging.info("[Server] Merging models from all clients.")
    global task_status
    model_files = [f for f in os.listdir() if f.startswith("client_model_shard_") and f.endswith(".pth")]
    if not model_files:
        logging.warning("[Server] No model files found for merging.")
        return {"message": "No models to merge"}

    models = [torch.load(f, map_location="cpu") for f in model_files]
    merged_model = models[0]
    for key in merged_model.keys():
        for model in models[1:]:
            merged_model[key] += model[key]
        merged_model[key] /= len(models)

    merged_model_filename = "merged_model.pth"
    torch.save(merged_model, merged_model_filename)
    logging.info(f"[Server] Models merged and saved as {merged_model_filename}")
    task_status = 0
    return {"message": "Models merged successfully", "merged_model_path": merged_model_filename}