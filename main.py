import os.path

import torch
from torch import tensor
from torch import concat
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import dataset
import affinity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
korte = dataset.KorteRaw('Korte', 60, (1920,1080),device=device)
model = affinity.AffinityNetwork(grid_size=60, image_size=(1920, 1080))

if os.path.exists("affinity.bin"):
    print("Reloading checkpoint from disk")
    model.load_state_dict(torch.load('affinity.bin'))

loss_fn = affinity.AffinityLoss(60, (1920, 1080))
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
train_loader = DataLoader(korte, batch_size=8, shuffle=True, num_workers=0)

print("The model will be running on", device, "device")
model.to(device)

try:
    for epoch in range(100):
        for i, (images, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Current loss: {loss.item()}                                                                                  ", end='\r')

except KeyboardInterrupt:
    pass

torch.save(model, "affinity.bin")