import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Normalization
from tensorflow.keras.optimizers import Adam
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
import numpy as np


class MultiLabelModel:
    def __init__(self, input_dim, output_dim, lr=1e-4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.model = self._build_model()

    def _build_model(self):
        norm_layer = Normalization()

        model = Sequential([
            Input(shape=(self.input_dim,)),
            norm_layer,
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(self.output_dim, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=self.lr)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.CosineSimilarity(name='cosine_sim'),
            ]
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100):
        validation = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation,
            epochs=epochs
        )
        return history

    def evaluate(self, X_test, y_test, threshold=0.5):

        y_pred_prob = self.model.predict(X_test)

        y_pred = (y_pred_prob >= threshold).astype("float32")
        y_true = y_test.astype("float32")

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)


        cos = tf.keras.losses.CosineSimilarity(reduction="none")
        cosine_sim = -cos(y_true, y_pred).numpy().mean()

        return {
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "cosine_similarity": cosine_sim
        }


    def save_model(self, filepath):
        self.model.save(filepath)

    def save_history(self, history, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(history.history, f)


    @staticmethod
    def load(filepath):

        model = load_model(filepath)

        input_dim = model.input_shape[1]
        output_dim = model.output_shape[1]

        instance = MultiLabelModel(input_dim, output_dim)

        instance.model = model
        return instance
