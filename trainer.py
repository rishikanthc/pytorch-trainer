import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.DataLoader as DataLoader

class trainer:
    def __init__(self, train_set, val_set = None, test_set, model, optimizer, lossfunc,
                 device = 'cpu', verbose, batchsize = 256):

        print("=> Configuring parameters")
        self.batch_size = batchsize
        self.train_set = train_set

        if val_set is not None:
            self.val_set = val_set

        self.test_set = test_set
        self.model = model
        self.optimizer = optimizer
        self.lossfunc = lossfunc
        self.deivce = device
        self.verbose = verbose

    def createDataloaders(self):

        print("=> Initializing dataloaders")
        self.train_loader = DataLoader(self.train_set, batch_size = self.batch_size,
                                       shuffle = True)
        self.val_loader = DataLoader(self.val_set, batch_size = self.batch_size,
                                       shuffle = True)
        self.test_loader = DataLoader(self.test_set, batch_size = self.batch_size,
                                       shuffle = True)

    def fit(self, epochs = 50, liveplot = False, early_stop = True, es_epochs=5):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        for epoch in range(epochs):
            train_epoch_loss = 0.0
            train_epoch_acc = 0.0

            for idx, datum in enumerate(self.train_loader):
                data, labels = datum[0].to(self.device), datum[1].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.lossfunc(output, labels)
                train_epoch_loss += loss.item()
                # _, preds = torch.max(output.data, 1)
                loss.backward()
                self.optimizer.step()

            train_epoch_loss /= idx
            val_epoch_loss, val_epoch_acc = self.validate(self.val_loader)
            train_loss.append(train_epoch_loss)

    def eval(self, data_loader):
        cum_loss = 0.0
        cum_acc = 0.0

        with torch.no_grad():
            for idx, datum in enumerate(self.data_loader):
                data, labels = datum[0].to(self.device), datum[1].to(self.device)
                output = self.model(data)
                loss = self.lossfunc(output, labels)
                cum_loss += loss.item()
                _, preds = torch.max(output.data, 1)
                cum_acc += (preds == labels).sum().item()

            cum_loss /= idx
            cum_acc /= idx

        return cum_loss, cum_acc

    def test(self):
        return self.eval(self.test_loader)

