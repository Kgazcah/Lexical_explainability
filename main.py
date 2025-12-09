import pandas as pd
import numpy as np
import utils

problem = 'software_requirements/no_stopwords'
df = pd.read_csv(f'data/{problem}/dataset.csv')
n_gram_n = 1


# preprocessing the dataset
if problem.split('/')[1] == 'stopwords':
    preprocessed_df = utils.preprocessing(df, 'basic', 'en')
else:
    preprocessed_df = utils.preprocessing(df, 'plus', 'en')

df_preprocessed = pd.DataFrame({'text': preprocessed_df})
df_preprocessed['class'] = df['class']
df_preprocessed.to_csv(f'assets/method/{problem}/df_preprocessed.csv', index=False)    

vocab_to_index= utils.get_vocab_to_index(preprocessed_df, output_file=f'assets/method/{problem}')
vocab_size = len(vocab_to_index)

top_words, vocab_pareto = utils.get_vocab_using_pareto(df_preprocessed, output_file=f'assets/method/{problem}')

print(f"Full vocabulary: {vocab_size}")
print(f"Reduced vocabulary using pareto: {len(vocab_pareto)}")
print(vocab_pareto)

