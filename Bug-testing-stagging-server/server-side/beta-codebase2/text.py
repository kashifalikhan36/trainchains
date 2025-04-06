import socket
import pickle
import os
from datasets import load_dataset

# Define paths and constants
HYPERPARAMS_FILE = "hyperparams.pkl"
SERVER_ADDRESS = ("0.0.0.0", 5000)
NUM_CLIENTS = 3

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("open-r1/OpenR1-Math-220k", split='train')
print("Dataset loaded.")

# Split the dataset into parts
print("Splitting dataset...")
split_datasets = dataset.train_test_split(test_size=1/NUM_CLIENTS)
print("Dataset split into parts.")

# Define the hyperparameter tuning function
def tune_hyperparameters(data):
    # Dummy hyperparameter tuning logic
    best_lr = 5e-5  # Example: find the best learning rate
    return {"learning_rate": best_lr}

# Define the training function
def train_model(data, hyperparams, model_name="EleutherAI/gpt-neo-125M"):
    import torch
    from transformers import GPTNeoForCausalLM, GPT2Tokenizer
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    # Load the model and tokenizer
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_datasets = data.map(tokenize_function, batched=True)

    # Create DataLoader
    train_dataloader = DataLoader(tokenized_datasets, batch_size=8, shuffle=True)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"])

    # Training loop
    model.train()
    for epoch in range(1):  # Local training epochs
        for batch in train_dataloader:
            inputs = batch['input_ids'].to('cuda')
            labels = batch['labels'].to('cuda')
            
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    return model.state_dict()

# Function to send data and code to a client
def send_data_and_code_to_client(client_socket, data, code):
    print("Sending data and code to client...")
    client_socket.sendall(pickle.dumps((data, code)))
    print("Data and code sent.")

# Function to receive data from a client
def receive_data_from_client(client_socket):
    print("Receiving data from client...")
    data = b""
    while True:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet
    print("Data received from client.")
    return pickle.loads(data)

# Check for existing hyperparameters
if os.path.exists(HYPERPARAMS_FILE):
    print("Loading existing hyperparameters...")
    with open(HYPERPARAMS_FILE, "rb") as f:
        hyperparams = pickle.load(f)
    print("Hyperparameters loaded:", hyperparams)
else:
    # Wait for clients to connect and perform hyperparameter tuning
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(SERVER_ADDRESS)
        server_socket.listen(NUM_CLIENTS)
        print("Server is waiting for clients to connect...")

        for i in range(NUM_CLIENTS):
            client_socket, _ = server_socket.accept()
            print(f"Client {i+1} connected.")

            # Send dataset and hyperparameter tuning code to client
            client_data = split_datasets['test'] if i == 0 else split_datasets['train']
            send_data_and_code_to_client(client_socket, client_data, tune_hyperparameters)

            # Receive best hyperparameters from client
            client_hyperparams = receive_data_from_client(client_socket)
            print(f"Received hyperparameters from client {i+1}: {client_hyperparams}")

            # Save the best hyperparameters
            if i == 0:  # For simplicity, use the first client's hyperparameters
                hyperparams = client_hyperparams
                with open(HYPERPARAMS_FILE, "wb") as f:
                    pickle.dump(hyperparams, f)
                print("Best hyperparameters saved.")

# Proceed with model training using the best hyperparameters
print("Starting model training with hyperparameters:", hyperparams)