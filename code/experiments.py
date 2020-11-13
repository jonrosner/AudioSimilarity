from CLAudio import CLAudio
from utils import get_data_split

################# CONFIGURATION PARAMETERS #################

model_type = "vggvox" # one of [vggvox, lstm, transformer]
pretrain_dataset_pattern = "../int16_pcm_10sec_parsed/**/*"
pretrain_dataset_data_type = "npx" # one of [npx, wav, flac, mp3]
embedding_dims = 1024
proj_head_shape = [1024,512]
pretrain_loss_function = "nt-xent" # one of [nt-xent, triplet]
pretrain_batch_size = 64
pretrain_num_epochs = 10
pretrain_optimizer = "adam" # one of [adam, sgd]

min_stretch_factor = 0.8
split_seconds = 5
sample_rate = 16000
augmentation_config = {
    "sample_rate": sample_rate,
    "original_duration": 10 * sample_rate,
    "split_duration": split_seconds * sample_rate,
    "min_stretch_duration": int(split_seconds * sample_rate * min_stretch_factor),
    "gain": [0.5, -6.0, 6.0],
    "whitenoise": [0.5, -40., -10],
    "highpass": [0.5, 1, 2000, 1, 2],
    "lowpass": [0.5, 1_000, 7_000, 1, 2],
    "pitchshift": [0.05, -3, 3],
    "timestretch": [0.05, min_stretch_factor, 1.3]
}

transfer_dataset = "voxceleb" # one of [voxceleb, music, birdsong]
transfer_dataset_path = "../voxceleb_train/wav"
transfer_dataset_idens_split = "../iden_split.txt"
transfer_dataset_data_type = "wav"
transfer_num_classes = 50 # -1 for all classes
transfer_num_samples = 20 # -1 for all samples
classification_head_shape = [1024, 256]
transfer_optimizer = "sgd" # one of [adam, sgd]
transfer_batch_size = 10
transfer_num_epochs = 100

############################################################

claudio = CLAudio(model_type, embedding_dims=embedding_dims, proj_head_shape=proj_head_shape, loss_function=pretrain_loss_function, optimizer=pretrain_optimizer)

print("###################### PRETRAIN MODEL ######################")
print(claudio.pretrain_model.summary())

# comment if you do not want to pretrain from scratch but load a model
#claudio.pretrain(pretrain_dataset_pattern, pretrain_dataset_data_type, augmentation_config, batch_size=pretrain_batch_size, num_epochs=pretrain_num_epochs)
claudio.load_pretrain_model("/media/joro/volume/Linux/uni/master-thesis/models/vgg-pretrained/model")

claudio.transfer(num_classes=transfer_num_classes, head_shape=classification_head_shape, optimizer=transfer_optimizer)

print("###################### TRANSFER MODEL ######################")
print(claudio.transfer_model.summary())

data_split = get_data_split(transfer_dataset, transfer_dataset_path, transfer_dataset_idens_split, n=transfer_num_classes, k=transfer_num_samples)

print(len(data_split["train"]))
print(len(data_split["test"]))
claudio.train_transfer("voxceleb", data_split["train"], transfer_dataset_data_type, batch_size=transfer_batch_size, num_epochs=transfer_num_epochs)
claudio.eval_transfer("voxceleb", data_split["test"], transfer_dataset_data_type)
