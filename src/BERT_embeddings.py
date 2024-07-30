import torch
from transformers import BertModel, BertTokenizer

class BERTEmbeddings:
    def __init__(self, model_name="dkleczek/bert-base-polish-uncased-v1"):

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_word_embeddings(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, padding=True, return_tensors="pt", truncation=True, max_length=512)
            # padding is used to make sure that the input is the same length as the model expects
            # which can be useful for example when we want to embed multiple sentences at once
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        return embeddings

    def get_sentence_embeddings_from_word_embeddings(self, word_embeddings, method="mean"):
        if method == "mean":
            return word_embeddings.mean(dim=1)
        elif method == "max":
            return word_embeddings.max(dim=1).values # max is returning both the max values and the indices of the max values. We only want the max values
        else:
            raise ValueError(f"Method {method} not supported. Please use 'mean' or 'max'.")

    def get_sentence_embeddings(self, text, method="mean"):
        word_embeddings = self.get_word_embeddings(text)
        sentence_embeddings = self.get_sentence_embeddings_from_word_embeddings(word_embeddings, method=method)
        return sentence_embeddings


        