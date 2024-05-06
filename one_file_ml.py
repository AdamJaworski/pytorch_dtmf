import torch.nn as nn
import torch
import librosa
import numpy as np

keys = {
    0: '',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '*',
    11: '0',
    12: '#'
}


class DTMFNet(nn.Module):
    def __init__(self):
        super(DTMFNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) #16
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) #32
        self.batch2 = nn.BatchNorm2d(64)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(64 * 128, 13)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        x = self.fc(x)
        return x


def decode_ctc(outputs):
    batch_size = outputs.size(0)
    decode = []
    output = outputs.detach()
    prev_idx = None
    for i in range(batch_size):
        _, idx = torch.max(output[i], dim=1)
        if idx == 0 or idx == prev_idx:
            prev_idx = idx
            continue
        decode.append(int(idx))
        prev_idx = idx

    return decode


def code_to_str(code):
    code_str = ''
    for key in code:
        code_str += keys[key]
    return code_str


def get_model_output(model: DTMFNet, input_tensor):
    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_tensor = output_tensor.log_softmax(2).permute(1, 0, 2)
        code_decoded = decode_ctc(output_tensor)
        code_str = code_to_str(code_decoded)
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