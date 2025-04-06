#!/usr/bin/env python3

import sys
import requests
import torch
from datasets import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader

def collate_fn(batch):
    # Simple collate function: just stack problem and answer
    problems = [item["problem"] for item in batch]
    answers = [item["answer"] for item in batch]
    return problems, answers

def main():
    # print("Downloading dataset 'open-r1/OpenR1-Math-220k'...")
    # dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train[:1%]")
    print("Dataset loaded. Preparing a simple model...")

    # A trivial model that tries to predict the length of the 'problem' text
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 1)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    dataloader = DataLoader(batch_size=8, shuffle=True, collate_fn=collate_fn)

    print("Starting a simple training loop for demonstration...")
    for epoch in range(1):
        for problems, answers in dataloader:
            # Convert problem length to tensor
            x = torch.tensor([[len(p)] for p in problems], dtype=torch.float)
            # Convert answer length to tensor (dummy target)
            y = torch.tensor([[len(a)] for a in answers], dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch complete. Loss: {loss.item():.2f}")

    print("Training demonstration complete. You could add GPU usage or more complex logic here.")

if __name__ == "__main__":
    main()
