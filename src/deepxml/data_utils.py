import os
import re
import numpy as np
import joblib
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.datasets import load_svmlight_file
from tqdm import tqdm
from typing import Iterable


def build_vocab(corpus: Iterable[Iterable[str]], 
                vocab_size=500000, 
                pad='<pad>', 
                unknown='<unk>',
                max_times=3, 
                freq_times=10):
    
    vocab = [pad, unknown]
    counter = Counter(token for t in corpus for token in t)

    for word, freq in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        if freq >= freq_times:
            vocab.append(word)

        if freq < max_times or vocab_size == len(vocab):
            break
    return np.asarray(vocab)


def convert_to_idx(texts, max_len=None, vocab=None, pad='<pad>', unknown='<unk>'):

    texts = [[vocab.get(word, vocab[unknown]) for word in line]
                        for line in tqdm(texts, desc='Converting token to id')]
    
    return truncate_text(texts, max_len, vocab[pad], vocab[unknown])


def truncate_text(texts, max_len=500, padding_idx=0, unknown_idx=1):
    if max_len is None:
        return texts
    texts = np.asarray([list(x[:max_len]) + [padding_idx] * (max_len - len(x)) for x in texts])
    texts[(texts == padding_idx).all(axis=1), 0] = unknown_idx
    return texts


def get_mlb(mlb_path, labels=None) -> MultiLabelBinarizer:
    if os.path.exists(mlb_path):
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def get_sparse_feature(feature_file, label_file):
    sparse_x, _ = load_svmlight_file(feature_file, multilabel=True)
    return normalize(sparse_x), np.load(label_file) if label_file is not None else None


def output_res(output_path, name, scores, labels):
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, F'{name}-scores'), scores)
    np.save(os.path.join(output_path, F'{name}-labels'), labels)


class Tokenizer:
    def __init__(self, language = "russian", max_len: int = 2048) -> None:
        self.language = language
        self.stop_words = stopwords.words(language) + ["это"]
        self.vocab = {}
        self.max_len = max_len

    def build_vocab(self, corpus: Iterable[str]):
        corpus = [[token.lower() for token in word_tokenize(text) if len(re.sub(r'[^\w]', '', token)) > 0] 
                  for text in corpus]
        
        corpus = [[i for i in doc if i not in self.stop_words] for doc in corpus]

        corpus = [[i for i in doc if not i.isdigit()] for doc in corpus]

        vocab = build_vocab(corpus)

        self.vocab = {word: i for i, word in enumerate(vocab)}

    def __call__(self, texts: Iterable[str]):

        texts = [[token.lower() for token in word_tokenize(text) if len(re.sub(r'[^\w]', '', token)) > 0] 
                  for text in texts]
        
        texts = [[i for i in doc if i not in self.stop_words] for doc in texts]

        texts = [[i for i in doc if not i.isdigit()] for doc in texts]


        return convert_to_idx(texts, self.max_len, self.vocab)
        
