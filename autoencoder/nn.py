import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
from autoencoder.losses import VlaLoss, FlaLoss, LacsLoss
# from tensorflow.keras.utils import plot_model
import pickle
import pandas as pd

class Autoencoder:
    def __init__(self, input_neurons=33, input_size=33, embedding_size=200,
                 optimizer='adam', metrics=['CosineSimilarity'], vocab_size=None, 
                 n_gram=1, bits_per_token=11, loss='fla', initial_weights_file=None): #tf.keras.losses.mean_squared_logarithmic_error          #'adam'
        """
        loss: fla (fixed logarithm absolute loss just for n_gram of size 1), lacs (logarithm absolute with cosine similarity), vla (variable logarithm absolute)
        
        """
        #hyperparameters
        self.optimizer = optimizer
        self.metrics = metrics
        self.n_gram = n_gram
        self.size = self.n_gram * bits_per_token
        self.losst = loss
        if self.losst == 'fla':
          self.loss = FlaLoss(vocab_size=vocab_size)
          print("ENTRÓ AQUÍ FLA")
        elif self.losst == 'lacs':
          self.loss = LacsLoss(vocab_size=vocab_size, total_bits=self.size, bits_p_word=bits_per_token, alpha=0.8) #with cosine
        elif self.losst == 'vla':
          self.loss = VlaLoss(vocab_size=vocab_size, n_gram=self.n_gram, bits_per_token=bits_per_token)
        self.initializer = tf.keras.initializers.GlorotUniform(seed=0)

        #create the nn
        self.autoencoder = tf.keras.models.Sequential()
        self.autoencoder.add(tf.keras.layers.Dense(input_neurons, input_shape=(input_size,)))
        self.autoencoder.add(tf.keras.layers.Dropout(0.3, seed=0))

        # encoder
        encoder = tf.keras.layers.Dense(
            embedding_size, 
            activation=tf.nn.sigmoid, 
            kernel_initializer=self.initializer
        )
        self.autoencoder.add(encoder)

        # decoder
        decoder = tf.keras.layers.Dense(
            input_size, 
            activation=tf.nn.sigmoid, 
            kernel_initializer=self.initializer
        )
        self.autoencoder.add(decoder)

        #save the layers indexes to encode and decode later
        self.index_last_encoder_layer = self.autoencoder.layers.index(encoder)
        self.index_decoder_layer = self.autoencoder.layers.index(decoder)
      
        if initial_weights_file is not None:
            with open(initial_weights_file, "rb") as f:
                weights = pickle.load(f)
            self.autoencoder.set_weights(weights)
            print(f"Loading weights from {initial_weights_file}")

        self.autoencoder.summary()

    def save_initialize_weights(self, initialize_weights_file='assets'):
        weights = self.autoencoder.get_weights()
        with open(initialize_weights_file, "wb") as f:
            pickle.dump(weights, f)

    def fit(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=256, shuffle=False):
        self.autoencoder.compile(optimizer=self.optimizer, metrics=self.metrics, loss=self.loss)
        history = self.autoencoder.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=(X_val, y_val)
        )
        return history
    
    def save(self, name='model.h5'):
        self.autoencoder.save(name)

# #ommit cobj for msle , custom_objects={"KariAbsLoss": KariAbsLoss}
#     def load_model(self, name='model.h5', vocab_size=None):
#         # self.autoencoder = load_model(name, custom_objects={"CustomLoss": lambda: CustomLoss(vocab_size=vocab_size)}) #, custom_objects={"KariAbsLoss": KariAbsLoss}
#         self.autoencoder = load_model(name, custom_objects={"CustomLoss": lambda **kwargs: CustomLoss(vocab_size=vocab_size, **kwargs)})

#         return self.autoencoder
    
    def load_model(self, name='model.h5', vocab_size=None, bits_per_token=None):
        if self.losst == 'lacs':
        # loading model with absolute and cosinesimilarity
          self.autoencoder = load_model(
              name,
              custom_objects={
                  "LacsLoss": lambda **kwargs: LacsLoss(
                      vocab_size=vocab_size,
                      bits_p_word=bits_per_token,
                      total_bits=self.size,
                      **kwargs
                  )
              }
          )
        elif self.losst == 'vla':
          #loading model only with absolute
          self.autoencoder = load_model(
              name,
              custom_objects={
                  "VlaLoss": lambda **kwargs: VlaLoss(
                      vocab_size=vocab_size,
                      n_gram=self.n_gram,
                      bits_per_token=bits_per_token,
                      **kwargs
                  )
              }
          )
        elif self.losst == 'fla':
           self.autoencoder = load_model(
              name,
              custom_objects={
                  "FlaLoss": lambda **kwargs: FlaLoss(
                      vocab_size=vocab_size,
                      **kwargs
                  )
              }
          )
        return self.autoencoder

   
    def predict(self, X):
        y_pred = self.autoencoder.predict(X)
        return y_pred

    def encode(self):
        self.encoder = Model(
            inputs=self.autoencoder.input, 
            outputs=self.autoencoder.get_layer(index=self.index_last_encoder_layer).output
        )
        return self.encoder

    def decode(self):
        self.decoder = Model(
            inputs=self.autoencoder.get_layer(index=self.index_last_encoder_layer).output, 
            outputs=self.autoencoder.get_layer(index=self.index_decoder_layer).output
        )
        return self.decoder