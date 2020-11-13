import numpy as np
import random
from scipy import signal
import pyrubberband as pyrb

def crop(a, n):
    """
    Crop n random samples from a signal a.
    """
    start = random.randint(0, len(a) - n - 1)
    return a[start:start+n]

def apply_gain(a, db):
    """
    Apply gain of db to a signal a.
    """
    gain_float = 10 ** (db / 20)
    return np.clip(a * gain_float, -1, 1)
    
def add_whitenoise(a, db):
    """
    Add white noise scaled by db to a signal a.
    """
    return np.clip(a + apply_gain(np.random.rand(len(a)) * 2 - 1, db), -1, 1)

def lowpass(a, cutoff, order, config):
    """
    Apply a butterworth low-pass filter at frequency cutoff to a signal a.
    """
    B, A = signal.butter(order, cutoff / (config["sample_rate"] / 2), btype="lowpass")
    return signal.lfilter(B, A, a, axis=0)

def highpass(a, cutoff, order, config):
    """
    Apply a butterworth high-pass filter at frequency cutoff to a signal a.
    """
    B, A = signal.butter(order, cutoff / (config["sample_rate"] / 2), btype="highpass")
    return signal.lfilter(B, A, a, axis=0)

def timestretch(a, factor):
    """
    Time-stretch a signal a by a factor.
    """
    return pyrb.time_stretch(a.astype(np.float32), 16000, factor)

# https://gist.github.com/cversek/1e355d961ba41535ff8296920586d9a3
def speedup(a, factor):
    """
    Speed up a signal a by a factor.
    """
    indices = np.round(np.arange(0, len(a), factor))
    indices = indices[indices < len(a)].astype(int)
    return a[indices]

def pitchshift(a, n, nfft=2048):
    """
    Pitch-shift a signal a by n semi-tones.
    """
    factor = 2**(1.0 * n / 12.0)
    stretched = timestretch(a, 1.0/factor)
    return speedup(stretched[nfft:], factor)

# combining it saves around 20% computational time
def timestretch_and_pitchshift(a, config, do_pitchshift, do_timestretch):
    """
    Apply time-stretch and pitch-shift at the same time.
    """
    if do_timestretch and not do_pitchshift:
        factor = random.uniform(config["timestretch"][1], config["timestretch"][2])
        return timestretch(a, factor)
    elif not do_timestretch and do_pitchshift:
        n = random.randint(config["pitchshift"][1], config["pitchshift"][2])
        return pitchshift(a, n)
    else:
        stretch_factor = random.uniform(config["timestretch"][1], config["timestretch"][2])
        pitch_n = random.randint(config["pitchshift"][1], config["pitchshift"][2])
        factor = 2**(1.0 * pitch_n / 12.0)
        stretched = timestretch(a, (1.0/factor) * stretch_factor)
        return speedup(stretched[2048:], factor)

def randomAudioAugmentation(input_samples_array, config):
    """
    Apply a random audio augmentation to a signal using an augmentation config.
    """
    start = random.randint(0, len(input_samples_array) - config["split_duration"] - 1)
    a = input_samples_array[start:start+config["split_duration"]]
    augmentation_applied = False
    if random.uniform(0, 1) < config["gain"][0]:
        augmentation_applied = True
        gain_db = random.uniform(config["gain"][1], config["gain"][2])
        a = apply_gain(a, gain_db)
    if random.uniform(0, 1) < config["whitenoise"][0]:
        augmentation_applied = True
        wn_db = random.uniform(config["whitenoise"][1], config["whitenoise"][2])
        a = add_whitenoise(a, wn_db)
    if random.uniform(0, 1) < config["highpass"][0]:
        augmentation_applied = True
        cutoff_hp = random.randint(config["highpass"][1], config["highpass"][2])
        order_hp = random.randint(config["highpass"][3], config["highpass"][4])
        a = highpass(a, cutoff_hp, order_hp, config)
    if random.uniform(0, 1) < config["lowpass"][0]:
        augmentation_applied = True
        cutoff_lp = random.randint(config["lowpass"][1], config["lowpass"][2])
        order_lp = random.randint(config["lowpass"][3], config["lowpass"][4])
        a = lowpass(a, cutoff_lp, order_lp, config)
    do_pitchshift = random.uniform(0, 1) < config["pitchshift"][0]
    do_timestretch = random.uniform(0, 1) < config["timestretch"][0]
    if do_timestretch or do_pitchshift:
        augmentation_applied = True
        a = timestretch_and_pitchshift(a, config, do_pitchshift, do_timestretch)
    a = a[:config["min_stretch_duration"]]
    return a
