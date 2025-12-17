import pickle
import pandas as pd
import numpy as np
import utils
from encoding.doc2vec import Doc2VecEncoder
from nn.nn import MultiLabelModel
from sklearn.model_selection import train_test_split
from visualization.plotting import Visualization 

problem = 'software_requirements/no_stopwords'
df = pd.read_csv(f'data/{problem}/dataset.csv')
n_gram_n = 1


############### Step 1: preprocessing the dataset
if problem.split('/')[1] == 'stopwords':
    preprocessed_df = utils.preprocessing(df, 'basic', 'en')
else:
    preprocessed_df = utils.preprocessing(df, 'plus', 'en')

df_preprocessed = pd.DataFrame({'text': preprocessed_df})
df_preprocessed['class'] = df['class']
df_preprocessed.to_csv(f'assets/method/{problem}/df_preprocessed.csv', index=False)    


############### Step 2: Getting the full and reduced vocabulary
vocab_to_index= utils.get_vocab_to_index(preprocessed_df, output_file=f'assets/method/{problem}')
vocab_size = len(vocab_to_index)

vocab_pareto, vocab_list = utils.get_vocab_using_pareto(df_preprocessed, threshold=.80, output_file=f'assets/method/{problem}')

print(f"Full vocabulary: {vocab_size}")
print(f"Reduced vocabulary using pareto: {len(vocab_pareto)}")
# print(vocab_pareto)
# print(vocab_list)


################ Step 3: Labeling the text

words, binary_classes = utils.binary_class(df_preprocessed['text'], vocab_list)
df_preprocessed['words'] = words
df_preprocessed['binary_class'] = binary_classes.tolist()

df_preprocessed.to_csv(f'assets/method/{problem}/df_preprocessed.csv', index=False)  

################ Step 4: Encoding the sentences

encoder = Doc2VecEncoder(vector_size=100)

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

print(len(df_train), len(df_val), len(df_test))


X_train = utils.to_numpy_array(df_train['doc2vec_embedding'])
X_val   = utils.to_numpy_array(df_val['doc2vec_embedding'])
X_test  = utils.to_numpy_array(df_test['doc2vec_embedding'])

y_train = utils.to_numpy_array(df_train['binary_class'])
y_val   = utils.to_numpy_array(df_val['binary_class'])
y_test  = utils.to_numpy_array(df_test['binary_class'])

####################### Step 6: Training the neural network
model = MultiLabelModel(
    input_dim=X_train.shape[1],
    output_dim=y_train.shape[1]
)

history = model.train(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=200
)
# print(history.history)
model.save_history(history, 'assets/nn_model/history.pkl')
model.save_model('assets/nn_model/model.h5')

##################### Step 9: Visualizing the training plots
plot = Visualization()
plot.plotting_metric(history.history, 'cosine_sim', 'val_cosine_sim', path=f'assets/learning_graphs/software_requirements/simple_neural_network/no_stopwords', fig_name='Learning training')
plot.plotting_loss(history.history, 'loss', 'val_loss', path=f'assets/learning_graphs/software_requirements/simple_neural_network/no_stopwords', fig_name='Loss training')

####################### Step 7: Testing the model 
model = MultiLabelModel.load('assets/nn_model/model.h5')
results = model.evaluate(X_test, y_test)

print(results)


