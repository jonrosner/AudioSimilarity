import tensorflow as tf

def create_vggvox(embedding_dims, name="vggvox"):
    """
    Create the CNN-based VGG-Vox network.
    """
    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.layers.Conv2D(96, (7,7), strides=(2,2), padding="valid", kernel_regularizer=tf.keras.regularizers.L2(5e-4), name=f"{name}_conv1"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), name="mpool1"))
    model.add(tf.keras.layers.Conv2D(256, (5,5), strides=(2,2), padding="valid", kernel_regularizer=tf.keras.regularizers.L2(5e-4), name=f"{name}_conv2"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), name="mpool2"))
    model.add(tf.keras.layers.Conv2D(384, (3,3), strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.L2(5e-4), name=f"{name}_conv3"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.L2(5e-4), name=f"{name}_conv4"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.L2(5e-4), name=f"{name}_conv5"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D((5,3), strides=(3,2), name=f"{name}_mpool5"))
    model.add(tf.keras.layers.Conv2D(4096, (9,1), strides=1, kernel_regularizer=tf.keras.regularizers.L2(5e-4), padding="valid", name=f"{name}_fc6"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=[1,2], name=f"{name}_apool6")))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(embedding_dims, kernel_regularizer=tf.keras.regularizers.L2(5e-4), name=f"{name}_embeddings"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    return model
