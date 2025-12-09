from preprocessing.interface_builder import Builder
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import re
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
import spacy

class Preprocessing(Builder):
  """
  A class for basic text preprocessing operations including lowercasing,
  punctuation removal, whitespace trimming, stopword removal, and lemmatization.
  
  Attributes:
      corpus (pd.DataFrame): The input DataFrame containing text data.
      lemmatizer (WordNetLemmatizer): A lemmatizer from NLTK for reducing words to their base form.
  """
  def __init__(self, corpus):
    self._original_df = corpus.copy()
    self.reset()
    self.lemmatizer = WordNetLemmatizer()
  
  def reset(self):
    self.corpus = self._original_df.copy()
    return self

  #lowercase the text
  def build_lowercase_text(self, column):
    text = self.corpus[column].str.lower()
    self.corpus = list(text)

  def build_remove_punctuation(self):
    #removing also numbers and under scores
    self.corpus = [re.sub(r'[^A-Za-zÁÉÍÓÚáéíóúÑñ\s]', '', t.strip()) for t in self.corpus]
    return self

  #old function
  # def build_remove_punctuation(self):
  #   self.corpus = [re.sub(r'[^\w\s]', '', t.strip()) for t in self.corpus]
  #   return self

  #remove noisy words
  # def build_removing_stopwords(self, language):
  #   if language == 'en':
  #     stop_words = set(stopwords.words('english'))
  #   elif language == 'es':
  #     stop_words = set(stopwords.words('spanish'))
  #   self.corpus = [' '.join([word for word in sentence.split() if word not in stop_words])
  #                  for sentence in self.corpus]
  #   return self

  def build_removing_stopwords(self, language):
      if language == 'en':
          stop_words = set(stopwords.words('english'))
      elif language == 'es':
          stop_words = set(stopwords.words('spanish'))
      else:
          stop_words = set()

      stop_words.add('subject')
      self.corpus = [' '.join([word for word in sentence.split() if word.lower() not in stop_words])
                    for sentence in self.corpus]
      return self



  # #lemmatize words and return a list of lists of strings
  # def build_lemmatize(self, language):
  #   if language == 'en':
  #     self.corpus = [self.lemmatizer.lemmatize(word) for word in self.corpus]
  #   elif language == 'es':
  #     self.nlp = spacy.load("es_core_news_sm")
  #     self.corpus = [' '.join([token.lemma_ for token in self.nlp(sentence)]) for sentence in self.corpus]
  #   return self
  

  def build_lemmatize(self, language):
      if language == 'en':
          # Asumimos que cada elemento de self.corpus es una oración o texto
          self.corpus = [
              re.sub(r'\s+', ' ', ' '.join([self.lemmatizer.lemmatize(word) for word in sentence.split()])).strip()
              for sentence in self.corpus
          ]
      elif language == 'es':
          self.nlp = spacy.load("es_core_news_sm")
          self.corpus = [
              re.sub(r'\s+', ' ', ' '.join([token.lemma_ for token in self.nlp(sentence)])).strip()
              for sentence in self.corpus
          ]
      return self

  def getProduct(self): 
    return self.corpus