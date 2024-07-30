import pickle as pkl

class Word2VecEmbeddings:
    def __init__(self, model_path="../data/word2vec.model"):

        with open(model_path, 'rb') as f:
            self.model = pkl.load(f)

    def get_embeddings(self, text):
        return self.model[text]

    def get_sentence_embeddings_from_word_embeddings(word_embeddings, method="mean"):
        if method == "mean":
            return word_embeddings.mean(dim=1)
        elif method == "max":
            return word_embeddings.max(dim=1).values

    