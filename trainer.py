import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class trainer:
    def __init__(self, epochs=50, batchsize=256, liveplot=False, jupyter=False):
        self.epochs = epochs
        self.batch_size = batchsize
        self.liveplot = liveplot
        self.jupyter = jupyter

def validate(model, dataloader):
    model.eval()
    val_running_loss = 0.0
    val_runing_correct = 0

    for datum in dataloader:
        x, y = data
