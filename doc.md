
- **Azure Windows VM (High Computational CPU-Based):**  
  Run the **Server** role here.  
  The server is responsible for providing hyperparameters and receiving the aggregated model. Since it handles incoming HTTP requests and lightweight computations, it fits well on a high-performance CPU machine.

  **Command on Azure VM (Server):**  
  ```bash
  python federated_training_fastapi.py --role server --port 5000
  ```

- **Laptop (CPU-Based):**  
  Run the **Aggregator** role here.  
  The aggregator collects model updates from clients, performs the aggregation (which is typically CPU-bound), and then sends the integrated model back to the server.

  **Command on Laptop (Aggregator):**  
  ```bash
  python federated_training_fastapi.py --role aggregator --port 6000 --server_addr http://<AZURE_VM_PUBLIC_IP>:5000 --expected_clients 1
  ```
  Replace `<AZURE_VM_PUBLIC_IP>` with the actual public IP of your Azure VM.

- **PC with GPU:**  
  Run the **Client** role here.  
  The client fetches hyperparameters from the server, trains the model using its GPU, and then sends its trained model to the aggregator.

  **Command on GPU PC (Client):**  
  ```bash
  python federated_training_fastapi.py --role client --server_addr http://<AZURE_VM_PUBLIC_IP>:5000
  ```
  Replace `<LAPTOP_IP>` with your laptop’s IP and `<AZURE_VM_PUBLIC_IP>` with your Azure VM’s IP.

---

### Summary

- **Server (Azure VM):** High computational CPU → **Server Role**  
- **Laptop (CPU):** → **Aggregator Role**  
- **GPU PC:** → **Client Role (for GPU training)**

Each role runs in its own terminal window (or on its respective machine) so that they can communicate over the network. This distribution leverages your high computational CPU VM for central coordination while offloading heavy model training to the GPU PC.
