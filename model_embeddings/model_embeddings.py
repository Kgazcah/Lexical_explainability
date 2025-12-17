import tensorflow as tf


def cosine_sim(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.keras.losses.cosine_similarity(y_true, y_pred) * -1


class Doc2VecToAutoencoder:
    """
    Maps Doc2Vec embeddings to the latent space of a trained autoencoder
    and reconstructs the corresponding BoW representation via the decoder.
    """

    def __init__(
        self,
        doc_dim=100,
        z_dim=200,
        hidden_dim=256,
        alpha=0.7,
        optimizer="adam",
        metrics=[cosine_sim]
    ):
        self.doc_dim = doc_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.optimizer = optimizer
        self.metrics = metrics

        self.model = self._build_model()
        self.model.compile(
            optimizer=self.optimizer,
            loss=self._z_loss(),
            metrics=self.metrics
        )

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.doc_dim,), name="doc2vec_input")

        x = tf.keras.layers.Dense(self.hidden_dim, activation="relu")(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(self.hidden_dim, activation="relu")(x)

        z_hat = tf.keras.layers.Dense(
            self.z_dim,
            activation=None,
            name="predicted_z"
        )(x)

        return tf.keras.Model(inputs, z_hat, name="doc2vec_to_z")

    def _z_loss(self):
        alpha = self.alpha

        def loss(z_true, z_pred):
            z_true_n = tf.nn.l2_normalize(z_true, axis=1)
            z_pred_n = tf.nn.l2_normalize(z_pred, axis=1)

            cos = tf.reduce_mean(
                tf.reduce_sum(z_true_n * z_pred_n, axis=1)
            )

            mse = tf.reduce_mean(tf.square(z_true - z_pred))

            return alpha * (1.0 - cos) + (1.0 - alpha) * mse

        return loss

    def fit(
        self,
        X_train,
        Z_train,
        X_val=None,
        Z_val=None,
        epochs=200,
        batch_size=128
    ):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=20,
                restore_best_weights=True
            )
        ]

        return self.model.fit(
            X_train,
            Z_train,
            validation_data=(X_val, Z_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks
        )

    def predict_z(self, X):
        return self.model.predict(X)

    def reconstruct_bow(self, X_doc2vec, autoencoder, threshold=0.5):
        """
        Full pipeline:
        Doc2Vec -> predicted z -> decoder -> reconstructed BoW
        """
        z_hat = self.predict_z(X_doc2vec)
        bow_hat = autoencoder.decode(z_hat)
        bow_binary = (bow_hat > threshold).astype(int)
        return bow_binary

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={"loss": self._z_loss()}
        )
        return self.model
