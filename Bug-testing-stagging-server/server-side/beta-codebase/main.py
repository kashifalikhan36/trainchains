import argparse
import uvicorn
import logging

from globals import num_clients_needed
from routes import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Server for HPC & Federated Examples")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to listen on")
    parser.add_argument("--num_clients_gpu", type=int, default=0, help="Number of clients needed to complete task (for GPU devices). 0 means all clients")
    args = parser.parse_args()

    if args.num_clients_gpu != 0 and args.num_clients_gpu < 5:
        raise ValueError("At least 5 clients are required for federated distributed learning")

    # Assign the parsed value to the global variable
    logging.info(f"[Main] Setting num_clients_needed = {args.num_clients_gpu}")
    num_clients_needed = args.num_clients_gpu

    # Now run uvicorn with routes.app
    uvicorn.run(app, host=args.host, port=args.port)
    