import pickle
import pandas as pd
import numpy as np
import utils
from sklearn.model_selection import train_test_split
from visualization.plotting import Visualization
from model_embeddings.model_embeddings import Doc2VecToAutoencoder
from autoencoder.nn import Autoencoder

problem = 'software_requirements/no_stopwords'
df_preprocessed = pd.read_csv('assets/method/software_requirements/no_stopwords/df_preprocessed.csv')


################### Step 1: Dividing the dataset

df_train, df_test = train_test_split(
    df_preprocessed,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

df_train, df_val = train_test_split(
    df_train,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print(len(df_train), len(df_val), len(df_test))

X_train = utils.read_emb(df_train['doc2vec_embedding'])
X_val   = utils.read_emb(df_val['doc2vec_embedding'])
X_test  = utils.read_emb(df_test['doc2vec_embedding'])

y_train = utils.read_emb(df_train['autoencoder_embedding'])
y_val = utils.read_emb(df_val['autoencoder_embedding'])
y_test = utils.read_emb(df_test['autoencoder_embedding'])

autoencoder = Autoencoder()
"""
################### Step 2: Training the neural network
autoencoder.load(f'assets/autoencoder_model/{problem}/model_autoencoder.h5')

model = Doc2VecToAutoencoder(
    doc_dim=100,
    z_dim=200,
    alpha=0.7
)

history = model.fit(
    X_train, 
    y_train,
    X_val,
    y_val,
    epochs=200
)

model.save('assets/models/software_requirements/no_stopwords/model_embeddings.keras')

################### Step 3: Visualizing

plot = Visualization()
plot.plotting_metric(history.history, 'cosine_sim', 'val_cosine_sim', path=f'assets/learning_graphs/software_requirements/model_embeddings/no_stopwords', fig_name='Learning training')
plot.plotting_loss(history.history, 'loss', 'val_loss', path=f'assets/learning_graphs/software_requirements/model_embeddings/no_stopwords', fig_name='Loss training')
"""
################### Step 4: Loading the model to predict
model = Doc2VecToAutoencoder()
model.load('assets/models/software_requirements/no_stopwords/model_embeddings.keras')
binary_emb_reconstructed = model.reconstruct_bow(X_test, autoencoder, threshold=0.5)

vocab_pareto = pd.read_csv("assets/method/software_requirements/no_stopwords/vocab_pareto.csv")
vocab_pareto = vocab_pareto['word'].tolist()
print(df_test.head())
indx = 5
print(f"Real BoW: {df_test['binary_class'].iloc[indx]}")
print(f"Reconstructed BoW: {binary_emb_reconstructed[indx]}")
print(f"Actual words in the sentence: {df_test['words'].iloc[indx]}")
words_in_reconstructed_vector = utils.vector_to_words(vocab_pareto, binary_emb_reconstructed[indx])
print(f"Predicted words in the sentence: {words_in_reconstructed_vector}")


