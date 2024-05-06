import os
import sys
import torch
import librosa
import numpy as np
# import sounddevice as sd


class Dataset:
    sounds_list: list
    gt_list: list
    size: int

    def __init__(self, data_path):
        self.sounds_list = []
        self.label_list  = []
        self.fs = 0
        print("Creating data set...")
        for file in os.listdir(data_path):
            audio, self.fs = librosa.load(data_path + file)
            self.sounds_list.append(audio)
            num = file.split('.')[0].split('_')[0]
            self.label_list.append(int(num))

            for i in range(5):
                self.label_list.append(int(num))
                new_audio = self.mess_up_audio(audio)
                self.sounds_list.append(new_audio)

        if len(self.sounds_list) == len(self.label_list):
            print(f"Dataset initialized correctly, {len(self.sounds_list)} sounds!")
        else:
            print("Error while creating dataset")
            sys.exit()

    @staticmethod
    def mess_up_audio(audio):
        signal_power = np.mean(audio ** 2)

        noise_power_ratio = np.random.uniform(0.1, 0.3)
        noise_power = signal_power * noise_power_ratio

        # Generate white Gaussian noise
        mean_noise = 0
        std_noise = np.sqrt(noise_power)
        noise = np.random.normal(mean_noise, std_noise, len(audio))
        audio_noisy = (audio * np.random.uniform(0.75, 1)) + noise
        audio_noisy = np.clip(audio_noisy, -1, 1)

        return audio_noisy

    def get(self):
        audio_len   = np.random.randint(1, 20)
        input_numpy = np.array([])
        code        = []
        for i in range(audio_len):
            index = np.random.randint(0, len(self.sounds_list))
            sound = self.sounds_list[index]
            input_numpy = np.append(input_numpy, sound)
            num_to_code = self.label_list[index]
            if num_to_code != 0:
                code.append(num_to_code)

        melspec = librosa.feature.melspectrogram(y=input_numpy, sr=self.fs, n_fft=2048, hop_length=512, n_mels=128)
        melspec = librosa.power_to_db(melspec, ref=np.max)
        input_tensor = torch.tensor(melspec, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        code_tensor  = torch.tensor(code, dtype=torch.int32)
        return input_tensor, code_tensor, code



