import numpy as np
from scipy import signal

# from: https://github.com/v-iashin/VoxCeleb/blob/master/identification.ipynb
def enhance(samples):
    """
    Apply pre-emphasis, remove DC and add dither.
    """
    pre_emphasis = 0.97
    # preemphasis filter
    samples = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])
    # removes DC component of the signal
    samples = signal.lfilter([1, -1], [1, -0.99], samples)
    # dither
    dither = np.random.uniform(-1, 1, samples.shape)
    spow = np.std(samples)
    samples = samples + 1e-2 * spow * dither
    return samples

def create_spectrogram(samples):
    """
    Convert an array of samples to a spectrogram.
    """
    rate = 16000
    window = 'hamming'
    Tw = 25
    Ts = 10
    Nw = int(rate * Tw * 1e-3)
    Ns = int(rate * (Tw - Ts) * 1e-3)
    nfft = 2 ** (Nw - 1).bit_length()
    _, _, spec = signal.spectrogram(samples, rate, window, Nw, Ns, nfft, mode='magnitude', return_onesided=False)
    spec *= rate / 10
    mu = spec.mean(axis=1).reshape(512, 1)
    sigma = np.clip(spec.std(axis=1), a_min=1e-6, a_max=None).reshape(512, 1)
    spec = (spec - mu) / sigma
    return spec
