import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
from autoencoder.losses import BCE_CS
import pickle

def cosine_sim(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.keras.losses.cosine_similarity(y_true, y_pred) * -1


def bce_plus_cosine(alpha=0.2):
    bce = tf.keras.losses.BinaryCrossentropy()

    def loss(y_true, y_pred):
        bce_loss = bce(y_true, y_pred)

        y_true_n = tf.nn.l2_normalize(y_true, axis=1)
        y_pred_n = tf.nn.l2_normalize(y_pred, axis=1)

        cos_sim = tf.reduce_mean(
            tf.reduce_sum(y_true_n * y_pred_n, axis=1)
        )

        return bce_loss + alpha * (1.0 - cos_sim)

    return loss



class Autoencoder:
    def __init__(self, 
                 input_size=650, 
                 embedding_size=200,
                 optimizer='adam',
                 loss= tf.keras.losses.BinaryCrossentropy(),#bce_plus_cosine(alpha=0.2),#BCE_CS(alpha=0.7),#'binary_crossentropy',
                 metrics=[cosine_sim]):
        """
        - input_size: binary_class_vector (650)
        - embedding_size: (100–200)
        """

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.autoencoder = tf.keras.models.Sequential()

        self.autoencoder.add(
            tf.keras.layers.Dense(
                input_size,
                activation='relu',
                input_shape=(input_size,)
            )
        )
        self.autoencoder.add(tf.keras.layers.Dropout(0.3))

        encoder_layer = tf.keras.layers.Dense(
            embedding_size,
            activation='relu',
            name="embedding_layer"
        )
        self.autoencoder.add(encoder_layer)

        decoder_layer = tf.keras.layers.Dense(
            input_size,
            activation='sigmoid',
            name="decoder_output"
        )
        self.autoencoder.add(decoder_layer)


        self.index_last_encoder_layer = self.autoencoder.layers.index(encoder_layer)
        self.index_decoder_layer = self.autoencoder.layers.index(decoder_layer)

        self.autoencoder.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
        )

        self.autoencoder.summary()

    def fit(self, X_train, X_val=None, epochs=100, batch_size=256):
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        )

        history = self.autoencoder.fit(
            X_train, X_train,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[callback]
        )
        return history

    def save(self, name='autoencoder.h5'):
        self.autoencoder.save(name)

    def save_history(self, history, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(history.history, f)

    def encode(self, X):
        encoder = Model(
            inputs=self.autoencoder.input,
            outputs=self.autoencoder.layers[self.index_last_encoder_layer].output
        )
        return encoder.predict(X)


    def decode(self, Z):
        decoder = Model(
            inputs=self.autoencoder.layers[self.index_last_encoder_layer].output,
            outputs=self.autoencoder.layers[self.index_decoder_layer].output
        )
        return decoder.predict(Z)
  
    def predict(self, X):
        return self.autoencoder.predict(X)

    def load(self, name='autoencoder.h5'):
        self.autoencoder = load_model(
            name,
            custom_objects={
                "cosine_sim": cosine_sim
            }
        )

        # Reconstruir índices del encoder/decoder después de cargar
        for i, layer in enumerate(self.autoencoder.layers):
            if layer.name == "embedding_layer":
                self.index_last_encoder_layer = i
            if layer.name == "decoder_output":
                self.index_decoder_layer = i

        return self.autoencoder