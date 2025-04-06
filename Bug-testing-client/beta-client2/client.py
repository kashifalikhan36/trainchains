import socket
import pickle
import torch
import flwr as fl

# Function to receive data and code from the server
def receive_data_and_code_from_server(server_address):
    print("Connecting to server to receive data and code...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(server_address)
        data = b""
        while True:
            packet = s.recv(4096)
            if not packet:
                break
            data += packet
    print("Data and code received from server.")
    return pickle.loads(data)

# Function to send data to the server
def send_data_to_server(server_address, data):
    print("Sending data to server...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(server_address)
        s.sendall(pickle.dumps(data))
    print("Data sent to server.")

# Connect to the server and receive dataset and code
server_address = ("20.244.34.219", 5000)  # Change port for each client
dataset, tune_hyperparameters = receive_data_and_code_from_server(server_address)

# Execute the hyperparameter tuning function
print("Executing hyperparameter tuning...")
best_hyperparams = tune_hyperparameters(dataset)
print("Best hyperparameters found:", best_hyperparams)

# Send the best hyperparameters back to the server
send_data_to_server(server_address, best_hyperparams)

# Define Flower client
class GPTNeoClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in model_state_dict.items()]

    def set_parameters(self, parameters):
        params_dict = zip(model_state_dict.keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model_state_dict.update(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Re-run the training function if needed
        return self.get_parameters(), len(dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Implement evaluation logic if needed
        return 0.0, len(dataset), {}

# Start Flower client
print("Starting Flower client...")
fl.client.start_numpy_client(server_address="20.244.34.219:5000", client=GPTNeoClient())

