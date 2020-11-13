import tensorflow as tf
import numpy as np
import json
from read_audio import read_audio
from augmentations import randomAudioAugmentation, crop
from utils import fp_to_pcm, pcm_to_fp
from preprocess import enhance, create_spectrogram

def create_pretrain_dataset(file_pattern, augmentation_config, data_type, loss_function, model_type, input_shape, batch_size):
    """
    Create a pre-train dataset without labels.
    """
    augmentation_config_encoded = json.dumps(augmentation_config) # we have to encode it to pass it through the execution graph
    d = tf.data.Dataset.list_files(file_pattern)
    d = d.shuffle(buffer_size=10_000)
    if loss_function == "nt-xent":
        d = d.map(lambda filename: _create_input_pair(filename, augmentation_config_encoded, data_type, model_type, input_shape), num_parallel_calls=16, deterministic=False)
    elif loss_function == "triplet":
        d = d.batch(batch_size=2, drop_remainder=True) # we batch two files together, one used for anchor and positive, one for negative
        d = d.map(lambda filenames: _create_input_triplet(filenames, data_type, model_type, input_shape), num_parallel_calls=16, deterministic=False)
    d = d.batch(batch_size=batch_size, drop_remainder=True)
    d = d.prefetch(batch_size)
    return d

def create_transfer_dataset(filenames, num_labels, split, data_type, model_type, dataset, input_shape, batch_size):
    """
    Create a transfer dataset with labels.
    """
    d = tf.data.Dataset.list_files(filenames)
    d = d.shuffle(buffer_size=100000)
    d = d.map(lambda filename: _create_input_example(filename, input_shape, num_labels, split, data_type, model_type, dataset), num_parallel_calls=16, deterministic=False)
    if split == "train" or split == "validation":
        d = d.batch(batch_size=batch_size, drop_remainder=True)
        d = d.prefetch(batch_size)
    elif split == "test":
        if model_type == "vggvox":
            d = d.batch(1)
    return d

def _create_pair(path, augmentation_config_encoded, data_type, model_type):
    """
    Create a contrastive learning pair for nt-xent loss.
    """
    augmentation_config = json.loads(augmentation_config_encoded.numpy())
    path = path.numpy()
    data_type = data_type.numpy().decode()
    model_type = model_type.numpy().decode()
    y = read_audio(path, data_type)
    
    y1 = randomAudioAugmentation(y, augmentation_config)
    y1 = fp_to_pcm(enhance(y1))
    y2 = randomAudioAugmentation(y, augmentation_config)
    y2 = fp_to_pcm(enhance(y2))

    if model_type == "vggvox":
        y1_stft = np.expand_dims(create_spectrogram(y1)[:,:300], axis=-1)
        y2_stft = np.expand_dims(create_spectrogram(y2)[:,:300], axis=-1)
    elif model_type == "lstm":
        y1_stft = create_spectrogram(y1)[:,:300].T
        y2_stft = create_spectrogram(y2)[:,:300].T
    elif model_type == "transformer":
        y1_stft = np.expand_dims(create_spectrogram(y1)[:,:300].T, axis=-1)
        y2_stft = np.expand_dims(create_spectrogram(y2)[:,:300].T, axis=-1)
        
    return np.array([y1_stft, y2_stft]).astype(np.float32)

def _create_triplet(filenames, data_type, model_type):
    """
    Create a triplet for the triplet loss.
    """
    filenames = filenames.numpy()
    data_type = data_type.numpy().decode()
    model_type = model_type.numpy().decode()

    filename_p = filenames[0]
    filename_n = filenames[1]

    y = read_audio(filename_p, data_type)
    a = crop(y, 4*16_000)
    a = fp_to_pcm(enhance(a))
    p = crop(y, 4*16_000)
    p = fp_to_pcm(enhance(p))

    y = read_audio(filename_n, data_type)
    n = crop(y, 4*16_000)
    n = fp_to_pcm(enhance(n))

    if model_type == "vggvox":
        a_stft = np.expand_dims(create_spectrogram(a)[:,:300], axis=-1)
        p_stft = np.expand_dims(create_spectrogram(p)[:,:300], axis=-1)
        n_stft = np.expand_dims(create_spectrogram(n)[:,:300], axis=-1)
    elif model_type == "lstm":
        a_stft = create_spectrogram(a)[:,:300].T
        p_stft = create_spectrogram(p)[:,:300].T
        n_stft = create_spectrogram(n)[:,:300].T
    elif model_type == "transformer":
        a_stft = np.expand_dims(create_spectrogram(a)[:,:300].T, axis=-1)
        p_stft = np.expand_dims(create_spectrogram(p)[:,:300].T, axis=-1)
        n_stft = np.expand_dims(create_spectrogram(n)[:,:300].T, axis=-1)

    return np.array([a_stft, p_stft, n_stft]).astype(np.float32)

def _create_example(filename, num_labels, split, data_type, model_type, dataset):
    """
    Create a single example and label for classification using CCE loss.
    """
    filename = filename.numpy().decode()
    data_type = data_type.numpy().decode()
    num_labels = num_labels.numpy()
    model_type = model_type.numpy().decode()
    split = split.numpy().decode()
    datset = dataset.numpy().decode()

    if dataset == "voxceleb":
        speaker = filename.split("/")[-3]
        label = int(speaker[3:]) - 1 # there is no label 0
    elif dataset == "music":
        label = int(filename.split("/").split("_")[1])
    elif dataset == "birdsong":
        label = int(filename.split("/")[-3])
    
    one_hot_label = np.zeros(num_labels)
    one_hot_label[label] = 1
    
    y = read_audio(filename, data_type)
    # comment this if you do not want to learn with random crops
    if len(y) < 4*16_000:
        y = np.tile(y, 2)
    y = crop(y, 4*16_000)
    y = fp_to_pcm(enhance(y))
    stft = create_spectrogram(y)

    if split == "train" or split == "validation":
        if model_type == "vggvox":
            stft = np.expand_dims(stft[:,:300], axis=-1)
        elif model_type == "lstm":
            stft = stft[:,:300].T
        elif model_type == "transformer":
            stft = np.expand_dims(stft[:,:300].T, axis=-1)
    elif split == "test":
        if model_type == "vggvox":
            stft = np.expand_dims(stft, axis=-1)
        elif model_type == "lstm":
            stft = stft.T
            stft = [stft[i*300:(i+1)*300] for i in range(stft.shape[0] // 300)]
            stft = np.array(stft)
        elif model_type == "transformer":
            stft = stft.T
            stft = [stft[i*300:(i+1)*300] for i in range(stft.shape[0] // 300)]
            stft = np.expand_dims(np.array(stft), axis=-1)
    return stft, one_hot_label

def _create_input_pair(filename, augmentation_config_encoded, data_type, model_type, input_shape):
    [pair, ] = tf.py_function(func=_create_pair, inp=[filename, augmentation_config_encoded, data_type, model_type], Tout=[tf.float32])
    pair.set_shape(input_shape)
    # 0 is a dummy label, will not be used
    return pair, 0

def _create_input_triplet(filenames, data_type, model_type, input_shape):
    [triplet, ] = tf.py_function(func=_create_triplet, inp=[filenames, data_type, model_type], Tout=[tf.float32])
    triplet.set_shape(input_shape)
    # 0 is a dummy label, will not be used
    return triplet, 0

def _create_input_example(filename, input_shape, num_labels, split, data_type, model_type, dataset):
    [stft, label] = tf.py_function(func=_create_example, inp=[filename, num_labels, split, data_type, model_type, dataset], Tout=[tf.float32, tf.int16])
    stft.set_shape(input_shape)
    label.set_shape([num_labels])
    return stft, label
