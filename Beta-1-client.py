import os
import io
import time
import argparse
import logging
import torch
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import optuna
import json

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def local_hyperparam_search():
    """
    Fallback: local Optuna-based hyperparameter search if the server has none.
    Helps pick stable hyperparams to reduce NaNs.
    """
    def local_objective(trial: optuna.Trial) -> float:
        # Restrict learning rate range to reduce chance of NaNs
        lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
        epochs = trial.suggest_int("epochs", 1, 3)
        batch_size = trial.suggest_categorical("batch_size", [1, 2])
        max_length = trial.suggest_int("max_length", 64, 128)

        # Dummy score, higher = better
        dummy_score = 1.0 / (lr * epochs * batch_size * (max_length / 64))
        return dummy_score

    study = optuna.create_study(direction="maximize")
    # 5 minute local search was mentioned, but time=3 is in code; keep as-is.
    study.optimize(local_objective, timeout=3)
    return study.best_params

def post_local_hyperparams_to_server(server_addr: str, hyperparams: dict):
    """
    Submit local hyperparams to server if the server doesn't have any yet.
    """
    try:
        resp = requests.post(f"{server_addr}/submit_hyperparameters", json=hyperparams, timeout=30)
        if resp.status_code == 200:
            logging.info("[Client] Successfully submitted local hyperparams to server.")
        else:
            logging.warning(f"[Client] Server responded with status: {resp.status_code}")
    except Exception as e:
        logging.error(f"[Client] Could not submit local hyperparams: {e}")

def get_server_hyperparameters(server_addr: str):
    """
    Fetch best hyperparams from the server. Returns None if not available.
    """
    try:
        resp = requests.get(f"{server_addr}/get_hyperparameters", timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 503:
            logging.info("[Client] Hyperparameters not yet ready on the server.")
            return None
        else:
            logging.warning(f"[Client] Unknown response code: {resp.status_code}")
            return None
    except Exception as e:
        logging.error(f"[Client] Could not fetch hyperparameters: {e}")
        return None

def get_server_dataset_shard(server_addr: str, client_id: int, total_clients: int):
    """
    Request the dataset shard for this client from the server.
    """
    try:
        shard_resp = requests.get(
            f"{server_addr}/get_dataset_shard",
            params={"client_id": client_id, "total_clients": total_clients},
            timeout=60
        )
        if shard_resp.status_code == 200:
            if ("message" in shard_resp.json() and 
                shard_resp.json()["message"] == "No more dataset shards available. Training complete or all shards assigned."):
                logging.info("[Client] No more dataset shards available from server.")
                return {"message": "No more dataset shards available"}
            return shard_resp.json()
        else:
            logging.error(f"[Client] Shard request returned status: {shard_resp.status_code}")
            return None
    except Exception as e:
        logging.error(f"[Client] Failed to retrieve dataset shard: {e}")
        return None

def notify_server_of_gpu_start(server_addr: str):
    """
    Let server know that this client has a GPU, which can prune CPU-based hyperparam search.
    """
    try:
        resp = requests.get(f"{server_addr}/gpu_has_arrived", timeout=20)
        if resp.status_code != 200:
            logging.warning(f"[Client] GPU start notification returned {resp.status_code}")
    except Exception as e:
        logging.warning(f"[Client] GPU start notification failed: {e}")

def notify_server_of_gpu_finish(server_addr: str):
    """
    (Optional) Let the server know the GPU is done, if needed in your workflow.
    """
    try:
        # If server implements a slot system, you could call /gpu_slot_finish here
        pass
    except Exception as e:
        logging.warning(f"[Client] GPU finish notification failed: {e}")

def train(hyperparams: dict, problems: list, answers: list, server_addr: str, client_id: int, shard_id: int):
    """
    Training loop using a simple GPT-Neo. Attempts to mitigate NaNs by:
    1) Lower default LR
    2) Gradient clipping
    3) Skipping batches if NaN appears
    4) Checking for potential infinite/NaN loss via scaler
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[Client] Training on {device}...")

    # Notify the server that we have started GPU usage
    notify_server_of_gpu_start(server_addr)

    lr = hyperparams.get("lr", 5e-6)
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

    notify_server_of_gpu_finish(server_addr)

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)

    try:
        url = f"{server_addr}/upload_model"
        files = {"model": ("client_model.pth", buf.getvalue())}
        data = {"client_id": client_id, "shard_id": shard_id}

        # Use query parameters on the first attempt
        url_with_query = f"{url}?client_id={client_id}&shard_id={shard_id}"
        resp = requests.post(url_with_query, files=files, data=data)
        logging.info(f"[Client] Model upload response: {resp.text}")

        # Fallback: try again without query params if 422
        if resp.status_code == 422:
            resp = requests.post(url, files=files, data=data)
            logging.info(f"[Client] Model upload response after fix: {resp.text}")

    except Exception as e:
        logging.error(f"[Client] Model upload failed: {e}")

def run_client(server_addr: str, client_id: int, total_clients: int):
    """
    - Fetch or locally determine hyperparams
    - Request dataset shard
    - Train model, skipping any NaN batches
    - Upload finished model
    - Check for new tasks and repeat until no more tasks from server.
    """
    if not server_addr.startswith("http://") and not server_addr.startswith("https://"):
        server_addr = "http://" + server_addr

    hyperparams = get_server_hyperparameters(server_addr)
    if hyperparams is None:
        logging.info("[Client] Server has no hyperparams, performing local search...")
        local_params = local_hyperparam_search()
        logging.info(f"[Client] Found local hyperparams: {local_params}")

        post_local_hyperparams_to_server(server_addr, local_params)

        time.sleep(2)
        hyperparams = get_server_hyperparameters(server_addr)
        if hyperparams is None:
            logging.info("[Client] Still no hyperparams from server. Using local results.")
            hyperparams = local_params

    shard_data = get_server_dataset_shard(server_addr, client_id, total_clients)
    if not shard_data:
        logging.error("[Client] Failed to retrieve shard data, exiting task loop.")
        # break
    # if "message" in shard_data and shard_data["message"] == "No more dataset shards available":
    #     logging.info("[Client] No more tasks available from server. Exiting task loop.")
    #     break
    if "problems" not in shard_data or "answers" not in shard_data or "shard_id" not in shard_data:
        logging.error("[Client] Invalid shard data format. Exiting task loop.")
        # break

    problems = shard_data["problems"]
    answers = shard_data["answers"]
    shard_id = shard_data["shard_id"]

    train(hyperparams, problems, answers, server_addr, client_id, shard_id)

    logging.info("[Client] Training and model upload completed. Checking for more tasks...")
    time.sleep(5)

    # while True:
    #     shard_data = get_server_dataset_shard(server_addr, client_id, total_clients)
    #     if not shard_data:
    #         logging.error("[Client] Failed to retrieve shard data, exiting task loop.")
    #         break
    #     if "message" in shard_data and shard_data["message"] == "No more dataset shards available":
    #         logging.info("[Client] No more tasks available from server. Exiting task loop.")
    #         break
    #     if "problems" not in shard_data or "answers" not in shard_data or "shard_id" not in shard_data:
    #         logging.error("[Client] Invalid shard data format. Exiting task loop.")
    #         break

    #     problems = shard_data["problems"][0:30]
    #     answers = shard_data["answers"][0:30]
    #     shard_id = shard_data["shard_id"]

    #     train(hyperparams, problems, answers, server_addr, client_id, shard_id)

    #     logging.info("[Client] Training and model upload completed. Checking for more tasks...")
    #     time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client script that handles training and sends model to server.")
    parser.add_argument("--server_addr", required=True, help="Server address, e.g. http://<IP>:<Port>")
    parser.add_argument("--client_id", type=int, default=0, help="Numeric client index")
    parser.add_argument("--total_clients", type=int, default=2, help="Number of clients in total")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logging.error("[Client] No CUDA device, training cannot proceed.")
        exit(1)

    run_client(
        server_addr=args.server_addr,
        client_id=args.client_id,
        total_clients=args.total_clients
    )
