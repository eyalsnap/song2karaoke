import os

import torch
from scipy.io import wavfile


class MockLoss(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MockLoss, self).__init__()

    def forward(self, y1, y2):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        t = torch.abs(y1 - y2).float()

        save_path = r'output/output1.wav'
        # save_path = r'C:\Users\Eyal\Desktop\eyal\python\data\sample_data\sample_data\_nopr_slman__am_ate_shoma_\net_output.wav'
        if os.path.exists(save_path):
            os.remove(save_path)

        fs = 44100
        data = y2.cpu().numpy()
        wavfile.write(save_path, fs, data)

        return torch.mean(t)
