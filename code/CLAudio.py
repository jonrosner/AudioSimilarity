import numpy as np
import tensorflow as tf
from utils import fp_to_pcm, pcm_to_fp
from augmentations import randomAudioAugmentation
from vggvox import create_vggvox
from lstm import create_lstm
from transformer import create_transformer
from head import create_head
from loss_functions import nt_xent_loss, triplet_loss
from input_pipeline import create_pretrain_dataset, create_transfer_dataset

class CLAudio:
    def __init__(self, model_type, embedding_dims, proj_head_shape, loss_function="nt-xent", optimizer="adam"):
        assert model_type in ["vggvox", "lstm", "transformer"], f"{model_type} is not a supported model."
        assert loss_function in ["nt-xent", "triplet"], f"{loss_function} is not a supported loss function."
        assert optimizer in ["sgd", "adam"], f"{optimizer} is not a supported optimizer"

        self.loss_function = loss_function
        self.model_type = model_type
        self.embedding_dims = embedding_dims
        self.pretrain_model = self._init_pretrain_model(model_type, proj_head_shape, loss_function, optimizer)
        
    def _init_pretrain_model(self, model_type, proj_head_shape, loss_function, optimizer):
        if loss_function == "nt-xent":
            pair_size = 2
        elif loss_function == "triplet":
            pair_size = 3
        
        if model_type == "vggvox":
            base_model = create_vggvox(self.embedding_dims)
            self.pretrain_input_shape = [pair_size,512,None,1]
        elif model_type == "lstm":
            base_model = create_lstm(self.embedding_dims)
            self.pretrain_input_shape = [pair_size,300,512]
        elif model_type == "transformer":
            base_model = create_transformer(self.embedding_dims)
            self.pretrain_input_shape = [pair_size,300,512,1]
        
        projection_head = create_head(proj_head_shape)

        input_samples = tf.keras.layers.Input(shape=self.pretrain_input_shape)

        if loss_function == "nt-xent":
            loss = nt_xent_loss
            [left_input, right_input] = tf.unstack(input_samples, num=2, axis=1)
            left = projection_head(base_model(left_input))
            right = projection_head(base_model(right_input))
            out = tf.stack([left, right])
        elif loss_function == "triplet":
            loss = triplet_loss
            [anchor_input, positive_input, negative_input] = tf.unstack(input_samples, num=3, axis=1)
            anchor_embeddings = projection_head(base_model(anchor_input))
            positive_embeddings = projection_head(base_model(positive_input))
            negative_embeddings = projection_head(base_model(negative_input))
            out = tf.stack([anchor_embeddings, positive_embeddings, negative_embeddings])
        model = tf.keras.Model(inputs=[input_samples], outputs=out)

        if optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        elif optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)
        
        model.compile(loss=loss, optimizer=opt)
        return model
    
    def _init_transfer_model(self, num_classes, head_shape, optimizer):
        self.num_classes = num_classes
        if self.model_type == "vggvox":
            base_model = create_vggvox(self.embedding_dims)
            self.transfer_input_shape = [512,None,1]
        elif self.model_type == "lstm":
            base_model = create_lstm(self.embedding_dims)
            self.transfer_input_shape = [300,512]
        elif self.model_type == "transformer":
            base_model = create_transformer(self.embedding_dims)
            self.transfer_input_shape = [300,512,1]
        
        classification_head = create_head(head_shape)
        classification_head.add(tf.keras.layers.Dense(num_classes, name="logits"))
        
        input_samples = tf.keras.layers.Input(shape=self.transfer_input_shape)

        model = tf.keras.Model(inputs=[input_samples], outputs=classification_head(base_model(input_samples)))

        if optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        elif optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)

        model.compile(optimizer=opt,
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy()])

        # transfer weights from pretrained to transfer and freeze them
        for i in range(len(self.pretrain_model.layers[2].layers)):
            model.layers[1].layers[i].set_weights(self.pretrain_model.layers[2].layers[i].get_weights())
            model.layers[1].layers[i].trainable = False
        return model
        
    def pretrain(self, file_pattern, data_type, augmentation_config, batch_size=80, num_epochs=10, save_dir=None):
        assert data_type in ["npx", "mp3", "flac", "wav"], f"{data_type} is not a supported datatype."
        dataset = create_pretrain_dataset(file_pattern, augmentation_config, data_type, self.loss_function, self.model_type, self.pretrain_input_shape, batch_size)
        self.history_callback = self.pretrain_model.fit(dataset, epochs=num_epochs)
        training_duration = time.time() - t1
        if save_dir:
            self.pretrain_model.save(os.path.join(save_dir, "model"))
    
    def load_pretrain_model(self, path):
        if self.loss_function == "nt-xent":
            model = tf.keras.models.load_model(path, custom_objects={"nt_xent_loss": nt_xent_loss, "tf": tf})
        elif self.loss_function == "triplet":
            self.pretrain_model = tf.keras.models.load_model(path, custom_objects={"triplet_loss": triplet_loss, "tf": tf})

    def save_pretrain_model(self, path):
        self.pretrain_model.save(path)
    
    def transfer(self, num_classes, head_shape, optimizer="sgd"):
        self.transfer_model = self._init_transfer_model(num_classes, head_shape, optimizer)
    
    def train_transfer(self, dataset, filenames, data_type, batch_size=50, num_epochs=10):
        assert dataset in ["voxceleb", "birdsong", "music"], f"{dataset} is not a supported dataset."
        gamma = 10 ** (np.log10(1e-4 / 1e-2) / (num_epochs - 1))
        def scheduler(epoch, lr):
            if epoch > 0:
                return lr * gamma
            else:
                return lr
        lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)
        dataset = create_transfer_dataset(filenames, self.num_classes, "train", data_type, self.model_type, dataset, self.transfer_input_shape, batch_size)
        self.transfer_model.fit(dataset, epochs=num_epochs, callbacks=[lrs])

    def eval_transfer(self, dataset, filenames, data_type):
        assert dataset in ["voxceleb", "birdsong", "music"], f"{dataset} is not a supported dataset."
        if self.model_type != "vggvox":
            shape = [None] + self.transfer_input_shape
        else:
            shape = self.transfer_input_shape
        dataset = create_transfer_dataset(filenames, self.num_classes, "test", data_type, self.model_type, dataset, shape, -1)

        if self.model_type == "vggvox":
            self.transfer_model.evaluate(dataset)
        else:
            # score over average of predictions
            test_acc = tf.keras.metrics.CategoricalAccuracy()
            test_acc_top_k = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
            for stfts, label in dataset:
                preds = self.transfer_model.predict(stfts)
                pred = np.expand_dims(np.mean(preds, axis=0), axis=0)
                label = np.expand_dims(label, axis=0)
                test_acc.update_state(label, pred)
                test_acc_top_k.update_state(label, pred)
            top1_result = test_acc.result().numpy()
            top5_result = test_acc_top_k.result.numpy()
            print("Top-1 accuracy: ", top1_result)
            print("Top-5 accuracy: ", top5_result)

    def load_transfer_model(self, path):
        self.transfer_model = tf.keras.models.load_model(path)
    
    def save_transfer_model(self, path):
        self.transfer_model.save(path)
