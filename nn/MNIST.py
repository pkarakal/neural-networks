from torch.nn import Module, Linear, Dropout
from torch.nn import functional as f


class MNIST(Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout_rate, output):
        super(MNIST, self).__init__()
        self.input_size = input_size
        self.fc1 = Linear(input_size, hidden_size1)
        self.fc2 = Linear(hidden_size1, hidden_size2)
        self.fc3 = Linear(hidden_size2, output)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = f.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

