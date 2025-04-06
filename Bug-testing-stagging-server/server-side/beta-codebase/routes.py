
import os
import io
import time
import json
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from datasets import load_dataset
import torch

# Import all the global variables
from globals import (
    server_dataset,
    best_hyperparams,
    hyperparams_file,
    gpu_has_arrived_event,
    stop_cpu_search_event,
    hyperparam_lock,
    cpu_pruned_once,
    task_status,
    num_clients_needed,
    clients_completed_task,
    dataset_shards_assigned,
    lock,
    device_assignments,
    big_model_shards,
    model_shards_assigned,
    split_model_lock,
    clients_completed_model_shards,
    shards_count
)

# Import the functions for hyperparameter searches and merging
from tasks import background_wait_and_optimize
from tasks import run_optuna_search
from merging import merge_models

logging.info("[Server] Initializing FastAPI app...")
app = FastAPI()

@app.on_event("startup")
def startup_event():
    global server_dataset
    global best_hyperparams
    global num_clients_needed
    global task_status
    logging.info("[Server] Loading dataset at startup...")
    server_dataset = load_dataset("open-r1/OpenR1-Math-220k")
    logging.info("[Server] Dataset loaded successfully.")

    if os.path.exists(hyperparams_file):
        with open(hyperparams_file, "r") as f:
            best_hyperparams = json.load(f)
        logging.info("[Server] Hyperparameters loaded from file.")

    if best_hyperparams is None:
        import threading
        thread = threading.Thread(target=background_wait_and_optimize, daemon=True)
        thread.start()
    else:
        logging.info("[Server] Hyperparams already available.")
    logging.info(f"[Server] Server started with num_clients_needed: {num_clients_needed}")

@app.get("/gpu_has_arrived")
def gpu_has_arrived():
    if not gpu_has_arrived_event.is_set():
        gpu_has_arrived_event.set()
        stop_cpu_search_event.set()
        logging.info("[Server] GPU arrived.")
    return {"message": "GPU acknowledged; CPU-based search will be stopped if running."}

@app.post("/submit_hyperparameters")
async def submit_hyperparameters(hyperparams: dict):
    global best_hyperparams
    best_hyperparams = hyperparams
    with open(hyperparams_file, "w") as f:
        json.dump(hyperparams, f, indent=2)
    logging.info(f"[Server] Received and saved hyperparameters: {hyperparams}")
    return {"message": "Hyperparameters received"}

@app.get("/get_hyperparameters")
def get_hyperparameters():
    global best_hyperparams
    if best_hyperparams is None:
        if os.path.exists(hyperparams_file):
            with open(hyperparams_file, "r") as f:
                best_hyperparams = json.load(f)
        else:
            raise HTTPException(status_code=503, detail="Hyperparameters not yet available.")
    return best_hyperparams

@app.get("/get_dataset_shard")
def get_dataset_shard(client_id: int, total_clients: int):
    global server_dataset
    global task_status
    global dataset_shards_assigned
    global num_clients_needed
    if not server_dataset or "train" not in server_dataset:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")
    if client_id < 0 or client_id >= total_clients:
        raise HTTPException(status_code=400, detail="Invalid client_id or total_clients.")
    train_data = server_dataset["train"]
    data_len = len(train_data)
    shard_size = data_len // total_clients
    with lock:
        shard_index = -1
        for i in range(total_clients):
            if not dataset_shards_assigned[i]:
                shard_index = i
                dataset_shards_assigned[i] = True
                break
        if shard_index == -1:
            return {"message": "No more dataset shards available. Training complete or all shards assigned."}
        start_idx = shard_index * shard_size
        end_idx = start_idx + shard_size
        if shard_index == total_clients - 1:
            end_idx = data_len
        subset = train_data.select(range(start_idx, end_idx))
        task_status = 1
        logging.info(f"[Server] Assigned shard {shard_index} to client {client_id}")
        return {"problems": subset["problem"], "answers": subset["answer"], "shard_id": shard_index}

@app.post("/upload_model")
async def upload_model(client_id: int, shard_id: int, model: UploadFile = File(...)):
    global task_status
    global clients_completed_task
    global dataset_shards_assigned
    global num_clients_needed
    contents = await model.read()
    try:
        _ = torch.load(io.BytesIO(contents), map_location="cpu")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load client model: {e}")
    filename = f"client_model_shard_{shard_id}_client_{client_id}_{int(time.time())}.pth"
    with open(filename, "wb") as f:
        f.write(contents)
    logging.info(f"[Server] Client {client_id} model for shard {shard_id} received and saved as {filename}")
    with lock:
        if client_id not in clients_completed_task:
            clients_completed_task.append(client_id)
        all_clients_trained = (len(clients_completed_task) >= num_clients_needed 
                               if num_clients_needed > 0 
                               else all(dataset_shards_assigned))
        if all_clients_trained:
            merge_result = merge_models()
            clients_completed_task.clear()
            dataset_shards_assigned[:] = [False] * len(dataset_shards_assigned)
            return {"message": "Client model received, models merged", "path": filename, "merge_status": merge_result}
        else:
            logging.info(f"[Server] Client {client_id} completed shard {shard_id}. Waiting for other clients. "
                         f"Clients completed: {len(clients_completed_task)}/{num_clients_needed if num_clients_needed > 0 else 0}")
            return {"message": f"Client model received for shard {shard_id}, waiting for other clients", "path": filename}

@app.get("/get_task_status")
def get_task_status_endpoint():
    global task_status
    return {"task_status": task_status}

@app.get("/get_assigned_gpu")
def get_assigned_gpu(client_id: int):
    global device_assignments
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        return {"device": "cpu"}
    if client_id not in device_assignments:
        device_assignments[client_id] = client_id % available_gpus
    return {"device": f"cuda:{device_assignments[client_id]}"}

@app.get("/split_big_model_for_distributed_learning")
def split_big_model_for_distributed_learning(number_of_shards: int):
    global big_model_shards
    global model_shards_assigned
    global split_model_lock
    global shards_count
    with split_model_lock:
        if number_of_shards < 5:
            raise HTTPException(status_code=400, detail="At least 5 shards are required for distributed model training.")
        shards_count = number_of_shards
        big_tensor = torch.randn(1024, 61440)
        total_size = big_tensor.numel()
        part_size = total_size // shards_count
        big_model_shards = []
        for i in range(shards_count):
            start_i = i * part_size
            end_i = (i + 1) * part_size if i != shards_count - 1 else total_size
            shard_slice = big_tensor.view(-1)[start_i:end_i].clone()
            big_model_shards.append(shard_slice)
        model_shards_assigned[:] = [False] * shards_count
        return {"message": "Big model is split for distributed training", "shards_count": shards_count}

@app.get("/get_model_shard")
def get_model_shard(client_id: int):
    global big_model_shards
    global model_shards_assigned
    global shards_count
    if not big_model_shards or shards_count == 0:
        raise HTTPException(status_code=400, detail="No big model is split yet.")
    shard_index = -1
    with split_model_lock:
        for i in range(shards_count):
            if not model_shards_assigned[i]:
                shard_index = i
                model_shards_assigned[i] = True
                break
        if shard_index == -1:
            return {"message": "No more model shards available. All assigned."}
        data_io = io.BytesIO()
        torch.save(big_model_shards[shard_index], data_io)
        return {"shard_index": shard_index, "model_shard": data_io.getvalue()}

@app.post("/upload_model_shard")
async def upload_model_shard(client_id: int, shard_index: int, shard_file: UploadFile = File(...)):
    global big_model_shards
    global model_shards_assigned
    global shards_count
    global clients_completed_model_shards
    contents = await shard_file.read()
    try:
        new_shard = torch.load(io.BytesIO(contents), map_location="cpu")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load shard: {e}")
    with split_model_lock:
        big_model_shards[shard_index] = new_shard
        if client_id not in clients_completed_model_shards:
            clients_completed_model_shards.append(client_id)
        if len(clients_completed_model_shards) >= shards_count:
            merged_tensor = torch.cat(big_model_shards)
            big_model_shards.clear()
            clients_completed_model_shards.clear()
            return {"message": "All shards uploaded, big model merged.", "size": merged_tensor.numel()}
        else:
            return {"message": "Shard received, waiting for others.", "uploaded_shards": len(clients_completed_model_shards)}

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.get("/get_client_script")
def get_client_script():
    code = """#!/usr/bin/env python3

import sys
import requests
import torch
from datasets import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader

def collate_fn(batch):
    # Simple collate function: just stack problem and answer
    problems = [item["problem"] for item in batch]
    answers = [item["answer"] for item in batch]
    return problems, answers

def main():
    print("Downloading dataset 'open-r1/OpenR1-Math-220k'...")
    dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train[:1%]")
    print("Dataset loaded. Preparing a simple model...")

    # A trivial model that tries to predict the length of the 'problem' text
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 1)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    print("Starting a simple training loop for demonstration...")
    for epoch in range(1):
        for problems, answers in dataloader:
            # Convert problem length to tensor
            x = torch.tensor([[len(p)] for p in problems], dtype=torch.float)
            # Convert answer length to tensor (dummy target)
            y = torch.tensor([[len(a)] for a in answers], dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch complete. Loss: {loss.item():.2f}")

    print("Training demonstration complete. You could add GPU usage or more complex logic here.")

if __name__ == "__main__":
    main()
"""
    return {"client_script": code}

"""lr = hyperparams.get("lr", 5e-6)
    epochs = hyperparams.get("epochs", 1)
    batch_size = hyperparams.get("batch_size", 1)
    max_length = hyperparams.get("max_length", 128)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    class SimpleDataset(Dataset):
        def __init__(self, p_list, a_list, tok, max_len):
            self.samples = [f"{p} {a}" for p, a in zip(p_list, a_list)]
            self.tokenizer = tok
            self.max_length = max_len

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.samples[idx],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            return {k: v.squeeze(0) for k, v in enc.items()}

    dataset_obj = SimpleDataset(problems, answers, tokenizer, max_length)
    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == 'nt' else 4
    )

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-08)
    scaler = GradScaler()

    total_steps = len(dataloader) * epochs
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for step, batch in enumerate(dataloader):
            step_count += 1
            inputs = batch["input_ids"].to(device)
            labels = inputs.clone()

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(inputs, labels=labels)
                loss = outputs.loss

            if not torch.isfinite(loss):
                logging.error("[Client] Loss is NaN or Inf. Skipping this batch.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            if step % 10 == 0:
                logging.info(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f}")
                steps_done = epoch * len(dataloader) + (step + 1)
                elapsed_time = time.time() - start_time
                avg_time_per_step = (elapsed_time / steps_done) if steps_done else 0
                steps_left = total_steps - steps_done
                est_time_left = steps_left * avg_time_per_step
                logging.info(f"[Client] Approx. remaining training time: {est_time_left:.2f}s")

        avg_epoch_loss = epoch_loss / max(1, step_count)
        logging.info(f"[Epoch {epoch}] Avg Loss: {avg_epoch_loss:.4f}")

    # If your workflow requires letting the server know you're done with the GPU:
    notify_server_of_gpu_finish(server_addr)

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)

    try:
        url = f"{server_addr}/upload_model"
        files = {"model": ("client_model.pth", buf.getvalue())}
        data = {"client_id": client_id, "shard_id": shard_id}
        resp = requests.post(url, files=files, data=data, timeout=60)
        logging.info(f"[Client] Model upload response: {resp.text}")
    except Exception as e:
        logging.error(f"[Client] Model upload failed: {e}")"""