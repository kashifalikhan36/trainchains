import os
import io
import time
import argparse
import logging
import torch
import torch.optim as optim
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

if not torch.cuda.is_available():
    print("Not compatible - No CUDA Based GPU Found")
    exit(1)

def local_hyperparam_search():
    def local_objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 1, 3)
        batch_size = trial.suggest_categorical("batch_size", [2, 4])
        max_length = trial.suggest_int("max_length", 64, 128)
        dummy_score = 1.0 / (lr * epochs * batch_size * (max_length / 64))
        return dummy_score

    study = optuna.create_study(direction="minimize")
    study.optimize(local_objective, timeout=1800)  # 30 minutes
    return study.best_params

def post_local_hyperparams_to_server(server_addr, hyperparams):
    try:
        resp = requests.post(f"{server_addr}/submit_hyperparameters", json=hyperparams, timeout=30)
        if resp.status_code == 200:
            logging.info("[Client] Successfully submitted local hyperparams to server.")
        else:
            logging.warning(f"[Client] Server returned status code: {resp.status_code}")
    except Exception as e:
        logging.error(f"[Client] Could not submit local hyperparams: {e}")

def get_server_hyperparameters(server_addr):
    try:
        resp = requests.get(f"{server_addr}/get_hyperparameters", timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 503:
            logging.info("[Client] Hyperparameters not yet ready on server.")
            return None
        else:
            logging.warning(f"[Client] Unknown response code: {resp.status_code}")
            return None
    except Exception as e:
        logging.error(f"[Client] Could not retrieve hyperparameters: {e}")
        return None

def get_server_dataset_shard(server_addr, client_id, total_clients):
    try:
        shard_resp = requests.get(
            f"{server_addr}/get_dataset_shard",
            params={"client_id": client_id, "total_clients": total_clients},
            timeout=30
        )
        return shard_resp.json()
    except Exception as e:
        logging.error(f"[Client] Failed to retrieve dataset shard: {e}")
        return None

def notify_server_of_gpu_start(server_addr):
    try:
        resp = requests.get(f"{server_addr}/gpu_has_arrived", timeout=20)
        if resp.status_code != 200:
            logging.warning(f"[Client] GPU start notification returned {resp.status_code}")
    except Exception as e:
        logging.warning(f"[Client] GPU start notification failed: {e}")

def notify_server_of_gpu_finish(server_addr):
    try:
        resp = requests.post(f"{server_addr}/gpu_slot_finish", timeout=20)
        if resp.status_code != 200:
            logging.warning(f"[Client] GPU finish notification returned {resp.status_code}")
    except Exception as e:
        logging.warning(f"[Client] GPU finish notification failed: {e}")

def train(hyperparams, problems, answers, server_addr):
    device = torch.device("cuda")
    logging.info(f"[Client] Training on {device}...")

    notify_server_of_gpu_start(server_addr)

    lr = hyperparams.get("lr", 2e-5)
    epochs = hyperparams.get("epochs", 2)
    batch_size = hyperparams.get("batch_size", 2)
    max_length = hyperparams.get("max_length", 128)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    class TextDataset(Dataset):
        def __init__(self, probs, ans, tok, max_len):
            self.texts = [p + " " + a for p, a in zip(probs, ans)]
            self.tokenizer = tok
            self.max_length = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            return {k: v.squeeze(0) for k, v in enc.items()}

    dataset_obj = TextDataset(problems, answers, tokenizer, max_length)
    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == 'nt' else 4
    )

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    total_steps = len(dataloader) * epochs
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            inputs = batch["input_ids"].to(device)
            labels = inputs.clone()

            with autocast(device_type='cuda'):
                outputs = model(inputs, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if step % 10 == 0:
                logging.info(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f}")

            steps_done = (epoch * len(dataloader)) + (step + 1)
            elapsed_time = time.time() - start_time
            avg_time_per_step = elapsed_time / steps_done if steps_done else 0
            steps_left = total_steps - steps_done
            remaining_time = steps_left * avg_time_per_step
            if step % 10 == 0:
                logging.info(f"[Client] Approx. remaining training time: {remaining_time:.2f}s")

        logging.info(f"[Epoch {epoch}] Loss: {epoch_loss / len(dataloader): .4f}")

    notify_server_of_gpu_finish(server_addr)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.getvalue()

def run_client(aggregator_addr: str, server_addr: str, client_id: int, total_clients: int):
    hyperparams = get_server_hyperparameters(server_addr)
    if hyperparams is None:
        logging.info("[Client] Server hyperparams not ready. Performing local hyperparam search.")
        local_params = local_hyperparam_search()
        logging.info(f"[Client] Local hyperparams found: {local_params}")
        post_local_hyperparams_to_server(server_addr, local_params)
        time.sleep(3)
        hyperparams = get_server_hyperparameters(server_addr)
        if hyperparams is None:
            logging.info("[Client] Still no hyperparams from server, using local hyperparams anyway.")
            hyperparams = local_params

    shard_data = get_server_dataset_shard(server_addr, client_id, total_clients)
    if not shard_data:
        logging.error("[Client] No shard data received. Exiting.")
        return

    problems = shard_data["problems"]
    answers = shard_data["answers"]
    model_bytes = train(hyperparams, problems, answers, server_addr)

    buffer = io.BytesIO(model_bytes)
    url = f"{aggregator_addr}/upload_client_model"
    files = {"model": ("client_model.pth", buffer.getvalue())}

    try:
        resp = requests.post(url, files=files, timeout=60)
        logging.info(f"[Client] Model upload response: {resp.text}")
    except Exception as e:
        logging.error(f"[Client] Failed to upload model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-based client with local hyperparam fallback")
    parser.add_argument("--aggregator_addr", required=True, help="Aggregator address http://<ip>:<port>")
    parser.add_argument("--server_addr", required=True, help="Server address http://<ip>:<port>")
    parser.add_argument("--client_id", type=int, default=0, help="Client index ID (0-based)")
    parser.add_argument("--total_clients", type=int, default=2, help="Total number of clients")
    args = parser.parse_args()

    run_client(
        aggregator_addr=args.aggregator_addr,
        server_addr=args.server_addr,
        client_id=args.client_id,
        total_clients=args.total_clients
    )
