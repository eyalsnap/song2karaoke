import torch.nn as nn
import torch.nn.functional as F


class MockNet(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MockNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=112, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=112, out_channels=112, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=112, out_channels=112, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv1d(in_channels=112, out_channels=112, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv1d(in_channels=112, out_channels=112, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv1d(in_channels=112, out_channels=112, kernel_size=1, stride=1, padding=0)
        self.conv7 = nn.Conv1d(in_channels=112, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        y_pred = x
        return y_pred
