from torch.nn import Module, Linear, Dropout
from torch.nn import functional as f


class MNIST(Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout_rate, output, hidden_size3=None,
                 hidden_size4=None):
        super(MNIST, self).__init__()
        self.input_size = input_size
        self.fc1 = Linear(input_size, hidden_size1)
        self.fc2 = Linear(hidden_size1, hidden_size2)
        self.fc3 = Linear(hidden_size2, output)
        self.fc4 = None
        self.fc5 = None
        self.dropout = Dropout(dropout_rate)
        if hidden_size3 is not None:
            self.fc3 = Linear(hidden_size2, hidden_size3)
            self.fc4 = Linear(hidden_size3, output)
        if hidden_size4 is not None:
            self.fc4 = Linear(hidden_size3, hidden_size4)
            self.fc5 = Linear(hidden_size4, output)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = f.relu(self.fc2(x))
        x = self.dropout(x)
        if self.fc4 is None:
            x = self.fc3(x)
        else:
            x = f.relu(self.fc3(x))
            x = self.dropout(x)
            if self.fc5 is None:
                x = self.fc4(x)
            else:
                x = f.relu(self.fc4(x))
                x = self.dropout(x)
                x = self.fc5(x)
        return x

