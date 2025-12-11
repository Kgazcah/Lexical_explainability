from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

class Doc2VecEncoder:
    def __init__(
        self,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=100
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None

    def fit(self, texts):
        tagged_docs = [
            TaggedDocument(words=text.split(), tags=[str(i)])
            for i, text in enumerate(texts)
        ]

        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs
        )

        self.model.build_vocab(tagged_docs)
        self.model.train(
            tagged_docs,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs
        )

    def transform(self, texts):
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        embeddings = [
            self.model.infer_vector(text.split())
            for text in texts
        ]
        return embeddings

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
