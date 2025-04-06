# Federated Learning Demo with CPU-Based Aggregator

This repository demonstrates a simple federated learning setup using FastAPI and PyTorch. The setup is divided into three roles:

1. server.py (CPU-based)
   • Loads a dataset.  
   • Provides hyperparameters and dataset shards to clients.  
   • Accepts a final aggregated model from the aggregator.  

2. aggregator.py (CPU-based)
   • Collects individual model states from each client.  
   • Aggregates them using federated averaging.  
   • Sends the integrated model back to the server.  
   • Does not require any GPU resources.  

3. client.py (GPU-based)  
   • Fetches hyperparameters and a dataset shard from the server.  
   • Trains locally on a CUDA-based GPU.  
   • Uploads the trained model to the aggregator.  

## Running the Setup

1. Install Python dependencies:  
   ```bash
   pip install fastapi uvicorn requests torch transformers datasets
   ```

2. Start the server (CPU-based):  
   ```bash
   python server.py --port 5000 --expected_clients 2
   ```
   This starts the FastAPI server on port 5000, expecting 2 client updates.

3. Start the aggregator (CPU-based):  
   ```bash
   python aggregator.py --port 6000 --server_addr http://localhost:5000 --expected_clients 2
   ```
   This starts the FastAPI aggregator on port 6000. It points to the server's address (localhost:5000) and expects 2 client updates.

4. Start each client (GPU-based). If you have two clients, for instance:  
   • Client 0:  
     ```bash
     python client.py \
       --aggregator_addr http://localhost:6000 \
       --server_addr http://localhost:5000 \
       --client_id 0 \
       --total_clients 2
     ```  
   • Client 1:  
     ```bash
     python client.py \
       --aggregator_addr http://localhost:6000 \
       --server_addr http://localhost:5000 \
       --client_id 1 \
       --total_clients 2
     ```

When training completes on each client, the aggregator receives their model weights, aggregates them once all clients have submitted, and finally sends the combined model to the server.

If no CUDA-based GPU is found on the client side, the script will print “Not compatible - No CUDA Based GPU Found” and exit.  
If the aggregator does not require a GPU, it simply runs on CPU.
