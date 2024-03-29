import os

from scipy.io import wavfile
import numpy as np


class AudioSample:

    def __init__(self, path):

        self.song_path = os.path.join(path, 'aligned_song_5_second.wav')
        self.karaoke_path = os.path.join(path, 'aligned_karaoke_5_second.wav')

    def get_audio_data(self):
        _, song = wavfile.read(self.song_path)
        _, prediction = wavfile.read(self.song_path)
        song = song.T
        prediction = prediction.T
        song = song.astype(np.float32)
        prediction = prediction.astype(np.float32)
        return prediction, song
