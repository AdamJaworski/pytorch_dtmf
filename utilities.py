import numpy as np
import torch
import matplotlib.pyplot as plt

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


def tensor_to_code(tensor) -> str:
    code = ''
    for array in tensor:
        array_ = array.detach().numpy()
        value = np.where(array_ == array_.max())[0][0]
        code += keys[value]
    return code


def display_freq(audio: np.ndarray, fs) -> None:
    fft_result = np.fft.fft(audio)
    fft_freq = np.fft.fftfreq(len(audio), 1 / fs)

    # Taking the magnitude of the FFT result (for volume) and only the first half (due to symmetry)
    n = len(fft_result) // 2
    fft_magnitude = np.abs(fft_result[:n]) * 2 / len(audio)

    x = 650  # Lower frequency limit
    y = 1500  # Upper frequency limit
    indices = np.where((fft_freq >= x) & (fft_freq <= y))[0]
    # Plotting the Frequency vs Volume (Amplitude) graph
    plt.figure(figsize=(14, 6))
    plt.plot(fft_freq[indices], fft_magnitude[indices])
    plt.xticks(np.arange(x, y, 50))
    plt.title('Frequency vs Volume (Amplitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Volume (Amplitude)')
    plt.grid(True)
    plt.show()


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
