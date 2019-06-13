from scipy.io import wavfile
from torch.utils.data import Dataset
import numpy as np
from data.audio_sample import AudioSample


class AudioDataset(Dataset):

    def __init__(self, dir_paths):

        self.audio_samples = [AudioSample(p) for p in dir_paths]

    def __len__(self):
        return len(self.audio_samples)

    def __getitem__(self, index):
        current_audio = self.audio_samples[index]
        prediction, song = current_audio.get_audio_data()
        return song, prediction
