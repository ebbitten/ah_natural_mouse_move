# from keras.callbacks import ModelCheckpoint
import data_loader as dl
import init_model
import plot_model
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
import numpy
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch

TRAIN = False



#
(inputs, paths, times) = dl.load_data("train/data.json")
# (test_inputs, test_paths) = dl.load_data("test/inputs.csv", "test/paths.json")

split = 20
test_inputs, train_inputs = dl.make_split(inputs, split)
test_paths, train_paths = dl.make_split(paths, split)
test_times, train_times = dl.make_split(times, split)
batch_size = 4


class CustomNumericDataSet(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.text[idx]
        sample = (data, label)
        return sample

test_set = CustomNumericDataSet(test_inputs, test_paths)
train_set = CustomNumericDataSet(train_inputs, train_paths)

trainloader = torch.utils.data.DataLoader(train_set)
testloader = torch.utils.data.DataLoader(test_set)

# Model
net = init_model.PathNet()
#
# file_path_best = "models/path/weights-improvement-{epoch:02d}-{val_acc:.2f}--hdf5"
file_path_best = "models/path/weights-improvement-adamh-test-path.pth"
file_path_state_dict = "models/path/path-state-dict.pth"
# checkpoint = ModelCheckpoint(file_path_best, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

# Fit
# history = model.fit(train_inputs, train_paths, epochs=1000, verbose=1, validation_split=0.05, callbacks=callbacks_list)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)

if TRAIN:
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i% 2000 == 1999:
                print (f'({epoch + 1}, {i+1:5d} loss {running_loss / 2000:.3f}')
                running_loss = 0.0
    torch.save(net, file_path_best)
else:
    net = torch.load(file_path_best)
    torch.save(net.state_dict(), file_path_state_dict)

size = 1
for x in range(size):
    values = numpy.array(train_inputs[x])
    paths = net(torch.Tensor(values.reshape(1, 2)))
    flatted = paths.flatten()

# Plot
# plot_model.plot(model, history, train_inputs, train_paths)
