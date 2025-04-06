import requests
import torch
import json
import io
import subprocess
import os

SERVER_URL = "http://20.244.34.219:5000"

def check_gpu():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

def notify_server_gpu_has_arrived():
    url = f"{SERVER_URL}/gpu_has_arrived"
    response = requests.get(url)
    print("Server response:", response.json())

def get_hyperparameters():
    url = f"{SERVER_URL}/get_hyperparameters"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Hyperparameters not yet available.")
        return None

def get_dataset_shard(client_id, total_clients):
    url = f"{SERVER_URL}/get_dataset_shard?client_id={client_id}&total_clients={total_clients}"
    response = requests.get(url)
    data = response.json()
    if "problems" in data and "answers" in data and "shard_id" in data:
        return data
    print("Shard not available or assigned. Details:", data)
    return None

def fetch_training_code():
    """
    Fetches the full training script from the server (GET /get_client_script)
    and writes it to 'server_training_script.py'.
    This script will use the local shard_data.json file when training.
    """
    url = f"{SERVER_URL}/get_client_script"
    response = requests.get(url)
    if response.status_code == 200:
        script_content = response.json().get("client_script", "")
        with open("server_training_script.py", "w", encoding="utf-8") as f:
            f.write(script_content)
        print("Fetched and saved the server training script: server_training_script.py")
    else:
        print("Failed to fetch the server training script.")
        script_content = None
    return script_content

def run_training_script():
    """
    Runs 'server_training_script.py' as a separate process.
    This script should read shard_data.json and produce a trained_model.pth file.
    """
    if not os.path.exists("server_training_script.py"):
        print("No training script found. Make sure fetch_training_code() ran first.")
        return
    subprocess.run(["python", "server_training_script.py"])

def upload_trained_model(client_id, shard_id):
    """
    Reads the 'trained_model.pth' file and uploads it to the server.
    """
    if not os.path.exists("trained_model.pth"):
        print("No trained_model.pth found. Make sure the training script produced it.")
        return

    with open("trained_model.pth", "rb") as model_file:
        model_data = model_file.read()

    files = {"model": ("trained_model.pth", model_data)}
    url = f"{SERVER_URL}/upload_model?client_id={client_id}&shard_id={shard_id}"
    response = requests.post(url, files=files)
    print("Upload response:", response.json())

def main():
    device = check_gpu()
    if device.startswith("cuda"):
        print("GPU is available on this client.")
        notify_server_gpu_has_arrived()
    else:
        print("No GPU found. Proceeding with CPU.")

    # Get the hyperparameters
    hyperparams = get_hyperparameters()
    if hyperparams is None:
        print("Hyperparams are not ready yet. Exiting.")
        return

    # Example client info
    client_id = 0
    total_clients = 5

    shard_info = get_dataset_shard(client_id, total_clients)
    if not shard_info:
        print("No shard assigned; nothing to train.")
        return

    # Extract the shard data
    shard_id = shard_info["shard_id"]
    problems = shard_info["problems"]
    answers = shard_info["answers"]

    # Save shard data to a JSON file so the server_training_script can use it
    shard_data = {
        "shard_id": shard_id,
        "problems": problems,
        "answers": answers,
        "hyperparams": hyperparams
    }
    with open("shard_data.json", "w", encoding="utf-8") as f:
        json.dump(shard_data, f, indent=2)

    # Now fetch and run the server's training script
    fetch_training_code()
    run_training_script()

    # Finally, upload the resulting trained model
    upload_trained_model(client_id, shard_id)

if __name__ == "__main__":
    main()