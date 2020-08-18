import matplotlib.pyplot as plt
import torch

class trainer:
    def __init__(self, train_set, test_set, val_set, model, optimizer, lossfunc,
                 device = 'cpu', verbose = True, batchsize = 256):

        print("=> Configuring parameters")
        self.batch_size = batchsize
        self.train_set = train_set

        if val_set is not None:
            self.val_set = val_set

        self.test_set = test_set
        self.model = model
        self.optimizer = optimizer
        self.lossfunc = lossfunc
        self.device = device
        self.verbose = verbose

    def createDataloaders(self):

        print("=> Initializing dataloaders")
        self.train_loader = torch.utils.DataLoader(self.train_set, 
                                                   batch_size = self.batch_size,
                                                   shuffle = True)
        self.val_loader = torch.utils.DataLoader(self.val_set,
                                                 batch_size = self.batch_size,
                                                 shuffle = True)
        self.test_loader = torch.utils.DataLoader(self.test_set,
                                                  batch_size = self.batch_size,
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
            val_epoch_loss, val_epoch_acc = self.eval(self.val_loader)

            if verbose:
                print(f'Epoch: {epoch:3d} Training Loss: {train_epoch_loss:.4f} \
                      Validation Loss: {val_epoch_loss:.4f}') 

            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

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

if __name__ == '__main__':
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    from torchvision.models import vgg16
    import torch.nn as nn
    import torch.optim as optim


    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    dataset = CIFAR10(root="./data/", train=True, download = True,
                      transform = transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])
    test_set = CIFAR10(root="./data/", train = False, download = True,
                      transform = transform)

    model = vgg16(pretrained = False)
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    criterion = nn.CrossEntropyLoss()

    trainer = trainer(train_set, val_set, test_set, model, optimizer, criterion)
