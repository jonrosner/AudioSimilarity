import numpy as np
from utils import fp_to_pcm, pcm_to_fp, buf_to_float
import soundfile as sf
from scipy.io import wavfile
import audioread
import scipy
import resampy

def read_audio(path, data_type):
    """
    Read different types of audio files form disk.
    """
    if data_type == "npx":
        with open(path, 'rb') as f:
            y = np.load(f)
        y = pcm_to_fp(y)
    elif data_type == "wav":
        _, y = wavfile.read(path)
        y = pcm_to_fp(y)
    elif data_type == "flac":
        y, _ = sf.read(path)
    elif data_type == "mp3":
        y = _load_audio(path)
    return y

def _load_mp3(filename):
    """
    Decode a mp3 file from disk.
    """
    y = []
    try:
        with audioread.audio_open(filename) as input_file:
            sr_native = input_file.samplerate
            n_channels = input_file.channels
            if sr_native != 44100 or n_channels != 2:
                return np.array([])

            for frame in input_file:
                frame = buf_to_float(frame)
                y.append(frame)

        y = np.concatenate(y)
        # reshape for stereo before parsing it to mono
        y = y.reshape((-1, n_channels)).T
        y = np.mean(y, axis=0)
        y = _resample(y, 44100, 16000)
        return y
    except Exception as e:
        print(filename, e)
        return np.array([])

# from: https://librosa.org/doc/0.7.2/_modules/librosa/core/audio.html
def _fix_length(data, size, axis=-1, **kwargs):
    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data

def _resample(y, orig_sr, target_sr, res_type='kaiser_best', fix=True, scale=False, **kwargs):
    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = _fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.asfortranarray(y_hat, dtype=y.dtype)
