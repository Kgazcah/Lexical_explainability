from abc import ABC, abstractmethod

#PreprocessorBuilder
class Builder(ABC):
  @abstractmethod
  def build_lowercase_text(self):
    pass

  @abstractmethod
  def build_remove_punctuation(self):
    pass

  @abstractmethod
  def build_removing_stopwords(self):
    pass

  @abstractmethod
  def build_lemmatize(self):
    pass


