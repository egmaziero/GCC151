import nltk
import unidecode
import string


class Preprocessing:

    def __init__(self):
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        self.stemmer = nltk.stem.RSLPStemmer()

    def remove_accents(self, text):
        return unidecode.unidecode(text)

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('','',string.punctuation))

    def tokenize_sentences(self, text):
        sentences = self.sent_tokenizer.tokenize(text)
        return sentences

    def tokenize_words(self, text):
        tokens = nltk.tokenize.word_tokenize(text)
        return tokens

    def lemmatize(self, text):
        return text

    def stemmize(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    def lowercase(self, text):
        return text.lower()
