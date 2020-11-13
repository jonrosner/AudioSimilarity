import tensorflow as tf

def create_transformer(embedding_dims, name="transformer"):
    """
    Creates the encoder part of a SpeechTranformer.
    """
    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.layers.Conv2D(64, 3, strides=2, padding="valid", kernel_regularizer=tf.keras.regularizers.L2(5e-4), activation="relu", name=f"{name}_conv1")) 
    model.add(tf.keras.layers.Conv2D(64, 3, strides=2, padding="valid", kernel_regularizer=tf.keras.regularizers.L2(5e-4), activation="relu", name=f"{name}_conv2")) 
    model.add(tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, x.shape[1], x.shape[2] * x.shape[3]]))) 
    model.add(tf.keras.layers.Dense(512, activation="relu", name=f"{name}_fc1"))
    model.add(PositionalEncoding(74, 512))
    for i in range(8):
        model.add(TransformerBlock(512, 8, 512))
    model.add(tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_gap"))
    model.add(tf.keras.layers.Dense(embedding_dims, kernel_regularizer=tf.keras.regularizers.L2(5e-4), activation="relu", name=f"{name}_fc"))
    return model

# from: https://keras.io/examples/nlp/text_classification_with_transformer/
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
  
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, pos, d_model):
      super(PositionalEncoding, self).__init__()
      self.pos_encodings = self.create_pos_encodings(pos, d_model)
    
    def create_pos_encodings(self, pos, d_model):
      position = tf.expand_dims(tf.range(pos, dtype=tf.float32), axis=1)
      i = tf.expand_dims(tf.range(d_model, dtype=tf.float32), axis=0)

      exponent = 2 * (i//2) / d_model
      angle_rates = 1 / tf.math.pow(10000, exponent)
      angle_rads = position * angle_rates

      # apply sin to even indices in the array; 2i
      evens = tf.math.sin(angle_rads[:, ::2])

      # apply cos to odd indices in the array; 2i+1
      odds = tf.math.cos(angle_rads[:, 1::2])

      evens_expanded = tf.expand_dims(evens, axis=2) 
      odds_expanded = tf.expand_dims(odds, axis=2)

      combined = tf.concat([evens_expanded, odds_expanded], axis=2)
      return tf.reshape(combined, [1, pos, d_model])
    
    def call(self, inputs):
      repeated_encodings = tf.repeat(self.pos_encodings, tf.shape(inputs)[0], axis=0)
      return inputs + repeated_encodings