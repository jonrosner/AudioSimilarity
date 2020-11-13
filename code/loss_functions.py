import tensorflow as tf

# from: https://github.com/google-research/simclr/blob/master/objective.py
def nt_xent_loss(y_true, y_pred):
    """
    The NT-Xent loss from the SimCLR paper
    """
    [x,v] = tf.unstack(y_pred, num=2)
    x = tf.math.l2_normalize(x, -1)
    v = tf.math.l2_normalize(v, -1)

    batch_size = tf.shape(x)[0]
    masks = tf.one_hot(tf.range(batch_size), batch_size)
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)

    logits_x_x = tf.matmul(x, x, transpose_b=True) / 0.1
    logits_x_x = logits_x_x - masks * 1e9

    logits_v_v = tf.matmul(v, v, transpose_b=True) / 0.1
    logits_v_v = logits_v_v - masks * 1e9

    logits_x_v = tf.matmul(x, v, transpose_b=True) / 0.1
    logits_v_x = tf.matmul(v, x, transpose_b=True) / 0.1

    loss_x = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_x_v, logits_x_x], 1))
    loss_v = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_v_x, logits_v_v], 1))

    loss = tf.reduce_mean(loss_x + loss_v)

    return loss

def triplet_loss(y_true, y_pred):
    """
    A basic triplet loss based on the hinge loss.
    """
    [a,p,n] = tf.unstack(y_pred, num=3)
    pos_dist = tf.reduce_sum((a - p)**2, axis=-1)
    neg_dist = tf.reduce_sum((a - n)**2, axis=-1)
    basic_loss = pos_dist - neg_dist + 0.1
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))               
    return loss
