import tensorflow as tf

def create_head(head_shape, name="head"):
    """
    Create a head-network, which is a simple fully-connected network.
    """
    head = tf.keras.Sequential(name=name)
    for i, shape in enumerate(head_shape):
        head.add(tf.keras.layers.Dense(shape, name=f"{name}_fc{i}"))
    return head