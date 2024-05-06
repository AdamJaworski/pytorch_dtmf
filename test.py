import torch
import librosa
import utilities
from model import DTMFNet
import numpy as np

wav_data = r'./Data/verified/'


def get_model_output(model: DTMFNet, input_tensor):
    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_tensor = output_tensor.log_softmax(2).permute(1, 0, 2)
        code_decoded = utilities.decode_ctc(output_tensor)
        code_str = utilities.code_to_str(code_decoded)
        print(code_str)


if __name__ == "__main__":
    model = DTMFNet()
    model.load_state_dict(torch.load('latest.pth', map_location='cpu'))

    data = r'challenge 2024.wav'
    audio, fs = librosa.load(data)
    melspec = librosa.feature.melspectrogram(y=audio, sr=fs, n_fft=2048, hop_length=512, n_mels=128)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    input_tensor = torch.tensor(melspec, dtype=torch.float32).unsqueeze(0)
    get_model_output(model, input_tensor)