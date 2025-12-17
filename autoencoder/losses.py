import tensorflow as tf

def cosine_sim(y_true, y_pred):
    """Métrica de similitud coseno"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return -tf.keras.losses.cosine_similarity(y_true, y_pred)

class BCE_CS:
    """Binary Crossentropy + Cosine Similarity loss"""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.bce = tf.keras.losses.BinaryCrossentropy()
        
    def __call__(self, y_true, y_pred):
        bce_loss = self.bce(y_true, y_pred)
        cos_dist = tf.keras.losses.cosine_similarity(y_true, y_pred)
        cos_loss = -tf.reduce_mean(cos_dist)
        return bce_loss + self.alpha * cos_loss
    
    def get_config(self):
        return {'alpha': self.alpha}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    # import tensorflow as tf

# class BCE_CS(tf.keras.losses.Loss):
#     def __init__(self, alpha=0.7, **kwargs):
#         super().__init__(**kwargs)
#         self.alpha = alpha
#         self.bce = tf.keras.losses.BinaryCrossentropy(
#             reduction=tf.keras.losses.Reduction.NONE
#         )

#     def reconstruction_loss(self, y_true, y_pred):
#         # BCE por muestra, promedio por batch
#         bce = self.bce(y_true, y_pred)
#         return tf.reduce_mean(bce)

#     def contrastive_loss(self, y_true, y_pred):
#         y_true_norm = tf.nn.l2_normalize(y_true, axis=1)
#         y_pred_norm = tf.nn.l2_normalize(y_pred, axis=1)
#         cos_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=1)
#         return tf.reduce_mean(1.0 - cos_sim)

#     def call(self, y_true, y_pred):
#         rec_loss = self.reconstruction_loss(y_true, y_pred)
#         cos_loss = self.contrastive_loss(y_true, y_pred)
#         return self.alpha * rec_loss + (1.0 - self.alpha) * cos_loss