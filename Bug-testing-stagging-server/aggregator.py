import os
import io
import time
import argparse
import logging
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import uvicorn
import threading

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

##############################
#       AGGREGATION CODE     #
##############################

def aggregate_models(state_dicts):
    """
    Simple federated averaging:
      For each key in the state dict, average the corresponding tensors.
      This aggregator is CPU-based; no GPU needed.
    """
    aggregated_state = {}
    for key in state_dicts[0].keys():
        aggregated_state[key] = torch.zeros_like(state_dicts[0][key], dtype=state_dicts[0][key].dtype)
    for state in state_dicts:
        for key in state.keys():
            aggregated_state[key] += state[key]
    for key in aggregated_state:
        aggregated_state[key] /= len(state_dicts)
    return aggregated_state


##############################
#      FASTAPI AGGREGATOR    #
##############################

client_models = []
expected_clients_global = None
server_addr_global = None

def run_aggregator(port: int, server_addr: str, expected_clients: int):
    """
    CPU-based aggregator:
      1) Receives model uploads from each client.
      2) Averages them.
      3) Sends the integrated model to the server.
    """
    global expected_clients_global, server_addr_global
    expected_clients_global = expected_clients
    server_addr_global = server_addr

    app = FastAPI()

    @app.post("/upload_client_model")
    async def upload_client_model(model: UploadFile = File(...)):
        global client_models
        contents = await model.read()

        # Load the client's state dict on CPU
        try:
            state_dict = torch.load(io.BytesIO(contents), map_location="cpu")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load client model: {e}")

        client_models.append(state_dict)
        logging.info(f"[Aggregator] Received client model ({len(client_models)}/{expected_clients_global})")

        # If all expected clients have sent models, start aggregation
        if len(client_models) >= expected_clients_global:
            threading.Thread(target=aggregate_and_send).start()

        return {"message": "Client model received"}

    def aggregate_and_send():
        global client_models, server_addr_global
        logging.info("[Aggregator] Aggregating client models...")

        aggregated_state = aggregate_models(client_models)

        buffer = io.BytesIO()
        torch.save(aggregated_state, buffer)
        buffer.seek(0)

        url = f"{server_addr_global}/upload_model"
        files = {"model": ("aggregated_model.pth", buffer.getvalue())}

        try:
            response = requests.post(url, files=files)
            logging.info(f"[Aggregator] Aggregated model sent to server: {response.json()}")
        except Exception as e:
            logging.error(f"[Aggregator] Failed to send aggregated model: {e}")

        client_models = []

    logging.info(
        f"[Aggregator] Starting CPU-based aggregator on port {port} expecting {expected_clients} clients.\n"
        f"Aggregated model will be sent to server at {server_addr_global}"
    )
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-based aggregator for Federated Training")
    parser.add_argument("--port", type=int, default=6000, help="Port for the aggregator")
    parser.add_argument("--server_addr", type=str, required=True, help="Server's address (http://<server_ip>:<port>)")
    parser.add_argument("--expected_clients", type=int, default=2, help="Expected client count")
    args = parser.parse_args()

    run_aggregator(args.port, args.server_addr, args.expected_clients)
