
import pickle
import pandas as pd
import numpy as np
import utils
from encoding.doc2vec import Doc2VecEncoder
from nn.nn import MultiLabelModel
from autoencoder.nn import Autoencoder
from sklearn.model_selection import train_test_split
from visualization.plotting import Visualization
from model_embeddings.model_embeddings import Doc2VecToAutoencoder

problem = 'software_requirements/no_stopwords'
df = pd.read_csv(f'data/{problem}/dataset.csv')
# n_gram_n = 1

############### Step 1: preprocessing the dataset
if problem.split('/')[1] == 'stopwords':
    preprocessed_df = utils.preprocessing(df, 'basic', 'en')
else:
    preprocessed_df = utils.preprocessing(df, 'plus', 'en')

df_preprocessed = pd.DataFrame({'text': preprocessed_df})
df_preprocessed['class'] = df['class']
# df_preprocessed.to_csv(f'assets/method/{problem}/df_preprocessed.csv', index=False)    


############### Step 2: Getting the full and reduced vocabulary
vocab_to_index= utils.get_vocab_to_index(preprocessed_df, output_file=f'assets/method/{problem}')
vocab_size = len(vocab_to_index)

vocab_pareto, vocab_list, vocab_pareto_ind = utils.get_vocab_using_pareto(df_preprocessed, threshold=.80, output_file=f'assets/method/{problem}')

print(f"Full vocabulary: {vocab_size}")
print(f"Reduced vocabulary using pareto: {len(vocab_pareto)}")

df_vocab_pareto = pd.DataFrame(list(vocab_pareto_ind.values()), columns=['word'])
df_vocab_pareto.to_csv("assets/method/software_requirements/no_stopwords/vocab_pareto.csv", index=False)


################ Step 3: Labeling the text

words, binary_classes = utils.binary_class(df_preprocessed['text'], vocab_list)
df_preprocessed['words'] = words
df_preprocessed['binary_class'] = binary_classes.tolist()

df_preprocessed.to_csv(f'assets/method/{problem}/df_preprocessed.csv', index=False)  

################ Step 4: Encoding the sentences

encoder = Doc2VecEncoder(vector_size=200)

embeddings = encoder.fit_transform(df_preprocessed['text'])

df_preprocessed['doc2vec_embedding'] = embeddings

df_preprocessed.to_csv(
    f'assets/method/{problem}/df_preprocessed.csv',
    index=False
)

################### Step 5: Dividing the text

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

# print(len(df_train), len(df_val), len(df_test))

# X_train = utils.to_numpy_array(df_train['doc2vec_embedding'])
# X_val   = utils.to_numpy_array(df_val['doc2vec_embedding'])
# X_test  = utils.to_numpy_array(df_test['doc2vec_embedding'])

X_all = utils.to_numpy_array(df_preprocessed['binary_class'])
X_train = utils.to_numpy_array(df_train['binary_class'])
X_val   = utils.to_numpy_array(df_val['binary_class'])
X_test  = utils.to_numpy_array(df_test['binary_class'])

####################### Step 6: Training the autoencoder

"""
epochs = 200

autoencoder = Autoencoder(input_size=X_train.shape[1], embedding_size=200)
history = autoencoder.fit(X_train, X_val, epochs=epochs)
# print(history.history)
autoencoder.save(f'assets/autoencoder_model/{problem}/model_autoencoder.h5')
autoencoder.save_history(history, f'assets/autoencoder_model/{problem}/history.pkl')

##################### Step 9: Visualizing the training plots

plot = Visualization()
plot.plotting_metric(history.history, 'cosine_sim', 'val_cosine_sim', path=f'assets/learning_graphs/software_requirements/autoencoder/no_stopwords', fig_name='Learning training')
plot.plotting_loss(history.history, 'loss', 'val_loss', path=f'assets/learning_graphs/software_requirements/autoencoder/no_stopwords', fig_name='Loss training')
"""
##################### Step 10: Encoding & Testing the model

#load the autoencoder
autoencoder = Autoencoder()
autoencoder.load(f'assets/autoencoder_model/{problem}/model_autoencoder.h5')
z_embeddings = autoencoder.encode(X_test)

reconstructed = autoencoder.decode(z_embeddings)
reconstructed_binary = (reconstructed > 0.5).astype(int)
indx = 0
print(f"Original binary class: {X_test[indx]}")
print(f"Embedding encoded for that binary embedding: {z_embeddings[indx]}")
print(f"Reconstructed binary class from the embedding: {reconstructed_binary[indx]}")

hamming = np.mean(X_test != reconstructed_binary, axis=1)

print("Hamming distance (mean):", round(hamming.mean(),3))

####################### Step 11:  Get all the embeddings

embeddings = autoencoder.encode(X_all)
print(embeddings)

####################### Step 12: Getting the dataset for training the embeddings

df_preprocessed['autoencoder_embedding'] = embeddings.tolist()

df_preprocessed.to_csv(
    f'assets/method/{problem}/df_preprocessed.csv',
    index=False
)

