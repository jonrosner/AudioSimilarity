import numpy as np
import voxceleb

def fp_to_pcm(data):
    return (data * (2**15 - 1)).astype(np.int16)

def pcm_to_fp(data):
    return data / (2**15 - 1)

def buf_to_float(x, n_bytes=2, dtype=np.float32):
    scale = 1./float(1 << ((8 * n_bytes) - 1))
    fmt = '<i{:d}'.format(n_bytes)
    return scale * np.frombuffer(x, fmt).astype(dtype)

def get_data_split(transfer_dataset, transfer_dataset_path, transfer_dataset_idens_split, n, k):
    if transfer_dataset == "voxceleb":
        return voxceleb.calculate_splits(transfer_dataset_path, transfer_dataset_idens_split, n=n, k=k)
    elif transfer_dataset == "music":
        return music.calculate_splits(transfer_dataset_idens_split, n=n, k=k)
    elif transfer_dataset == "birdsong":
        return birdsong.calculate_splits(transfer_dataset_idens_split, n=n, k=k)
