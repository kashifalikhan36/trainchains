import threading
import multiprocessing
import logging
import torch
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Global variables (unchanged from original)
server_dataset = None
best_hyperparams = None
hyperparams_file = "best_hyperparams.json"
gpu_has_arrived_event = threading.Event()
stop_cpu_search_event = threading.Event()
hyperparam_lock = threading.Lock()
cpu_pruned_once = False
task_status = 0
num_clients_needed = 0
clients_completed_task = []
dataset_shards_assigned = [False] * 100
lock = threading.Lock()
device_assignments = {}
big_model_shards = []
model_shards_assigned = []
split_model_lock = threading.Lock()
clients_completed_model_shards = []
shards_count = 0