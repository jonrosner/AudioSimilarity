import tensorflow as tf

def create_lstm(embedding_dims, name="lstm"):
    """
    A basic two-layer Bidirectional-LSTM network.
    """
    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True, input_shape=[300,512]), name=f"{name}_l1"))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dims, return_sequences=False), name=f"{name}_l2"))
    return model
