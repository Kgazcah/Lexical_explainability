import pandas as pd
import numpy as np
import ast
from collections import Counter
from preprocessing.interface_builder import Builder
from preprocessing.director import Director
from preprocessing.preprocessor_builder import Preprocessing
from vocabulary.getting_vocabulary import GettingVocabulary


#preprocessing the corpus (Builder Pattern)
def preprocessing(df, type, language='en'):
    director = Director()
    builder = Preprocessing(df)
    if type == 'basic':
        #basic builder includes stopwords
        preprocessed_df = director.makeBasicPreprocessing(builder, language)
    elif type == 'plus':
        #plus builder does not include stopwords
        preprocessed_df = director.makePlusPreprocessing(builder, language)
    return preprocessed_df


#Getting the vocabulary and their indexes
def get_vocab_to_index(preprocessed_df, output_file='assets/method'):
    vocab_obj = GettingVocabulary(preprocessed_df)
    vocabulary = vocab_obj.get_vocab()
    vocab = pd.DataFrame(vocabulary)
    vocab.to_csv(f'{output_file}/vocabulary.csv', index=False)

    #assign a decimal index to each word from the vocabulary
    #vocabulary to index document will have something as follows:
    #{'x': 1, ..., 'yet': 1984, 'zero': 1985}
    vocab_to_index = vocab_obj.get_vocab_to_indx()
    vocab_to_index_df = pd.DataFrame(list(vocab_to_index.items()), 
                                    columns=['word', 'index'])
    vocab_to_index_df.to_csv(f'{output_file}/vocab_to_index.csv', index=False)
    return vocab_to_index 

#Uploading the data
def upload_embeddings(file_name, column):
    df = pd.read_csv(file_name, dtype=str)
    return np.vstack(df[column].apply(lambda x: np.array(list(x), dtype=float)).values)


def get_vocab_using_pareto(
    df,
    text_col="text",
    threshold=.80,
    ascending=False,
    output_file=None
):
    """
    It gets the reduced vocab using pareto 
    """

    #joining all the sentences
    all_text = " ".join(df[text_col].astype(str))

    #tokenizing
    tokens = all_text.split()

    #getting the relative frequency
    counter = Counter(tokens)
    # print(f"Counter: {counter}")
    freq_df = pd.DataFrame(counter.items(), columns=["word", "frequency"])
    freq_df = freq_df.sort_values(by="frequency", ascending=ascending)
    freq_df = freq_df.reset_index(drop=True)

    total = freq_df["frequency"].sum()
    freq_df["relative_frequency"] = freq_df["frequency"] / total

    #getting the cumulative percentage
    freq_df["cumulative_frequency"] = freq_df["relative_frequency"].cumsum()

    if output_file:
        freq_df.to_csv(f"{output_file}/frequency.csv", index=False)

    result_df = freq_df[freq_df["cumulative_frequency"] < threshold]
    vocab_list = result_df["word"].to_list()
    vocab_pareto = result_df["word"].to_dict()
    swapped_vocab_pareto = {v: k for k, v in vocab_pareto.items()}
    
    return swapped_vocab_pareto, vocab_list


def binary_class(texts, vocab_list):
    vocab_index = {word: i for i, word in enumerate(vocab_list)}
    
    all_words_in_text = []
    all_classes = []

    for t in texts:
        words = t.split()

        #words in the vocab_list
        filtered = [w for w in words if w in vocab_index]
        all_words_in_text.append(filtered)

        #binary class
        vec = np.zeros(len(vocab_list), dtype=int)
        for w in filtered:
            vec[vocab_index[w]] = 1

        all_classes.append(vec)

    return all_words_in_text, np.array(all_classes)

def to_numpy_array(series):
    return np.array(series.tolist(), dtype=float)


#reading all type of embeddings
def parse_embedding(x):
    if isinstance(x, np.ndarray):
        return x.astype(float)

    if isinstance(x, list):
        return np.array(x, dtype=float)

    if not isinstance(x, str):
        raise ValueError(f"Tipo no soportado: {type(x)}")

    x = x.strip()

    try:
        return np.array(ast.literal_eval(x), dtype=float)
    except (ValueError, SyntaxError):
        pass

    return np.fromstring(x.strip("[]"), sep=" ")

def read_emb(series):
    return np.vstack(series.apply(parse_embedding))
