import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import torch
from torch.utils.data import DataLoader
import os

class trainer:
    def __init__(self, train_set, test_set, val_set, model, optimizer, lossfunc,
                 device = 'cpu', verbose = True, batchsize = 256, savepath='./model'):

        print("=> Configuring parameters")
        self.batch_size = batchsize
        self.train_set = train_set

        if val_set is not None:
            self.val_set = val_set

        self.test_set = test_set
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lossfunc = lossfunc
        self.device = device
        self.verbose = verbose
        self.best_model = self.model.state_dict()
        self.savepath = savepath

        self.createDataloaders()

    def createDataloaders(self):

        print("=> Initializing dataloaders")
        self.train_loader = DataLoader(self.train_set, batch_size = self.batch_size,
                                       shuffle = True)
        self.val_loader = DataLoader(self.val_set, batch_size = self.batch_size,
                                     shuffle = True)
        self.test_loader = DataLoader(self.test_set, batch_size = self.batch_size,
                                      shuffle = True)

    def fit(self, epochs = 50, liveplot = False, early_stop = True, es_epochs=5):
        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        epoch_arr = []
        best_loss = float('inf')
        es_counter = 0

        print("=> Beginning training")
        self.model.train()

        for epoch in range(epochs):
            train_epoch_loss = 0.0
            train_epoch_acc = 0.0
            epoch_arr.append(epoch)

            for idx, datum in enumerate(self.train_loader):
                data, labels = datum[0].to(self.device), datum[1].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.lossfunc(output, labels)
                train_epoch_loss += loss.item()
                _, preds = torch.max(output.data, 1)
                train_epoch_acc += (preds == labels).sum().item()
                loss.backward()
                self.optimizer.step()

            train_epoch_loss /= idx
            train_epoch_acc /= len(self.train_loader.dataset)
            val_epoch_loss, val_epoch_acc = self.eval(self.val_loader)

            if self.verbose:
                print(f'\tEpoch: {epoch:3d} Training Loss: {train_epoch_loss:.4f} Validation Loss: {val_epoch_loss:.4f} Training Acc: {train_epoch_acc: .3f} Validation Acc: {val_epoch_acc: .3f}') 

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                self.best_model = self.model.state_dict()

            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

            if (epoch > 0 and val_loss[-1] > val_loss[-2]):
                es_counter += 1
                if es_counter == es_epochs:
                    print(f'==> Early stopping at epoch {epoch}')
                    break
            else:
                es_counter = 0

            if liveplot:
                plt.cla()
                plt.plot(epoch_arr, train_loss, label='train')
                plt.plot(epoch_arr, val_loss, label='validation')
                plt.legend(loc='upper left')
                plt.pause(0.005)

        if liveplot:
            plt.tight_layout()
            plt.show()

        self.history = {'train loss': train_loss, 'train acc': train_acc,
                        'val loss': val_loss, 'val acc': val_acc}
        
        print("=> Saving model to file")
        if not os.path.isdir(self.savepath):
            os.mkdir(self.savepath)
        savepath = os.path.join(self.savepath, 'trained_model')
        torch.save(self.best_model, savepath)

        return self.history

    def eval(self, data_loader):
        cum_loss = 0.0
        cum_acc = 0.0

        self.model.eval()

        with torch.no_grad():
            for idx, datum in enumerate(data_loader):
                data, labels = datum[0].to(self.device), datum[1].to(self.device)
                output = self.model(data)
                loss = self.lossfunc(output, labels)
                cum_loss += loss.item()
                _, preds = torch.max(output.data, 1)
                cum_acc += (preds == labels).sum().item()

            cum_loss /= idx
            cum_acc /= len(data_loader.dataset)

        return cum_loss, cum_acc

    def test(self):
        print("=> Evaluating on test dataset")
        self.model.load_state_dict(self.best_model)
        _, test_acc = self.eval(self.test_loader)
        print(f'\t Test accuracy: {test_acc}')

    def genPlots(self, path='./plots/'):
        print("=> Generating plots")
        if not os.path.isdir(path):
            os.mkdir(path)

        epoch_arr = list(range(len(self.history['train acc'])))
        plt.ioff()
        fig = plt.figure()
        plt.plot(epoch_arr, self.history['train loss'], label = 'train')
        plt.plot(epoch_arr, self.history['val loss'], label = 'validation')
        plt.legend(loc = 'upper right')
        savepath = os.path.join(path, 'loss.png')
        plt.savefig(savepath)
        plt.close(fig)
        
        fig = plt.figure()
        plt.plot(epoch_arr, self.history['train acc'], label = 'train')
        plt.plot(epoch_arr, self.history['val acc'], label = 'validation')
        plt.legend(loc = 'upper right')
        savepath = os.path.join(path, 'acc.png')
        plt.savefig(savepath)
        plt.close(fig)

if __name__ == '__main__':
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    from torchvision.models import vgg16
    import torch.nn as nn
    import torch.optim as optim
    from models import VGG


    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    dataset = CIFAR10(root="./data/", train=True, download = True,
                      transform = transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])
    test_set = CIFAR10(root="./data/", train = False, download = True,
                      transform = transform)

    #model = vgg16(pretrained = False)
    model = VGG('VGG16')
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    criterion = nn.CrossEntropyLoss()

    trainer = trainer(train_set, val_set, test_set, model, optimizer, criterion,
                      device='cuda')
    history = trainer.fit(epochs = 10, liveplot = False, es_epochs = 5)
    trainer.test()
    trainer.genPlots()
