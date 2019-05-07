#%% Task 1: Loading data
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
import random


class SignDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        # Anything could go here, e.g. image loading from file or a different structure
        datapoint = self.data[index]
        target = self.target[index]
        transform = Compose([ToTensor()])
        return transform(datapoint), torch.tensor(target)

    def __len__(self):
        return len(self.data)


def integer_encoder(labels):
    values = np.array(labels)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    return integer_encoded


def load_images(train_dir='data/train', limit=100, target_size=[32,32]):
    class_folders = os.listdir(train_dir)
    train_images = []
    train_labels = []
    for folder in class_folders:
        path_folder = os.path.join(train_dir,folder)
        if os.path.isdir(path_folder):
            files = os.listdir(path_folder)
            number_of_files = len(files)
            for i in range(number_of_files):
                if not files[i].startswith('.'):
                    path = os.path.join(path_folder, files[i])
                    img = Image.open(path)
                    img = img.convert('RGB')  # some images are in grayscale
                    img = img.resize(target_size)
                    train_images.append(np.array(img))
                    train_labels.append(folder)
    return train_images, train_labels


images, char_labels = load_images()
labels = integer_encoder(char_labels)

batch_size = 32
train_size = int(len(images)*0.9)

random.Random(1).shuffle(images)
random.Random(1).shuffle(labels)

train_images = images[:train_size]
train_labels = labels[:train_size]

validation_images = images[train_size:]
validation_labels = labels[train_size:]

train_dataset = SignDataset(train_images, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = SignDataset(validation_images, validation_labels)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_images, test_char_labels = load_images('data/test')
test_labels = integer_encoder(test_char_labels)
test_dataset = SignDataset(test_images, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#%% Task 2: Apply the model (implement train and test method)
import torch
import torch.nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, optimizer, loss_fn):
    model.train()
    losses = []
    n_correct = 0
    for iteration, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        n_correct += torch.sum(output.argmax(1) == labels).item()
    accuracy = 100.0 * n_correct / len(train_loader.dataset)
    return np.mean(np.array(losses)), accuracy


def eval(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    n_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)
            test_loss += loss.item()
            n_correct += torch.sum(output.argmax(1) == labels).item()

    average_loss = test_loss / len(test_loader)
    accuracy = 100.0 * n_correct / len(test_loader.dataset)
    return average_loss, accuracy


def plot_loss(train_losses, val_losses, n_epochs):
    plt.figure()
    plt.plot(np.arange(n_epochs), train_losses)
    plt.plot(np.arange(n_epochs), val_losses)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('Train/Val loss')
    plt.show()


#%% Task 3
from torchvision.models.densenet import DenseNet
import torch.nn as nn

def fit(model, optimizer, loss_fn, n_epochs, train_dataloader, val_dataloader):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(n_epochs):
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, loss_fn)
        val_loss, val_accuracy = eval(model, val_dataloader, loss_fn)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(
            epoch + 1, n_epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy))

    return train_losses, train_accuracies, val_losses, val_accuracies


model_dense = DenseNet(num_classes=24)
model_dense = model_dense.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(model_dense.parameters(), lr=learning_rate,  betas=(0.9, 0.999), eps=1e-8)
n_epochs = 300
loss_fn = nn.CrossEntropyLoss()

train_losses_result, train_accuracies_result, val_losses_result, val_accuracies_result = fit(model_dense, optimizer, loss_fn, n_epochs, train_dataloader, val_dataloader)

#plot_loss(train_losses_result, val_losses_result, n_epochs)

torch.save(model_dense.state_dict(),"model_full.pt")

loss_fn = nn.CrossEntropyLoss()

test_loss_result, test_accuracy_result = eval(model_dense, test_dataloader, loss_fn)
print('Test loss: ' + str(test_loss_result) + ' and test accuracy: ' + str(test_accuracy_result))
