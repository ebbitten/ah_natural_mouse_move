# from keras import Sequential
# from keras.layers import Dense, Reshape
from torch import nn, Tensor, from_numpy
import torch

target_path_count = 100

class PathNet(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2,1000)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(1000,target_path_count*2)

    def forward(self, seq):
        l1 = self.l1(seq.float())
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        return torch.reshape(l3, (target_path_count, 2))
        # return l3


# def init_model_paths():
#     model = Sequential()
#     model.add(Dense(1000, activation='relu', input_dim=2))
#     model.add(Dense(target_path_count * 2, activation='linear'))
#     model.add(Reshape((target_path_count, 2)))
#
#     # model.summary()
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])
#
#     return model
class TimeNet(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.float()
        self.l1 = nn.Linear(2, 216)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(216, target_path_count)


    def forward(self, seq):
        l1 = self.l1(seq.float())
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        return l3

# def init_model_time():
#     model = Sequential()
#     model.add(Dense(216, activation='relu', input_dim=2))
#     model.add(Dense(target_path_count, activation='linear'))
#     model.add(Reshape((target_path_count, 1)))
#
#     # model.summary()
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])
#
#     return model
