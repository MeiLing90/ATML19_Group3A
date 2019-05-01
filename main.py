#%% Testing out plots generation
import matplotlib.pyplot as plt
import numpy as np
for i in range(100):
    b = np.array([[i*1, 255, 233], [1, i*2, 233], [1, 255, i*3]])
    plt.imshow(b)
    plt.show()

#%% Testing out something else
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([9, 1, 2, 1, 2, 1, 7, 8, 9])
plt.plot(x, y)
plt.show()

#%% Task 3: Apply the model (implement train and test method)
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


def test(model, test_loader, loss_fn):
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


def plot(train_losses, val_losses, epoch_n):
    plt.figure()
    plt.plot(np.arange(epoch_n), train_losses)
    plt.plot(np.arange(epoch_n), val_losses)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('Train/Val loss')
