
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from tqdm import tqdm

import numpy as np


class LDAEmbeddings:
    def __init__(self, n_topics, **lda_kwargs) -> None:
        self.n_topics = n_topics
        self.lda_kwargs = lda_kwargs

    def fit(self, texts):
        self.dictionary = Dictionary(texts)

        corpus = [self.dictionary.doc2bow(text) for text in texts]

        self.lda = LdaModel(corpus, num_topics=self.n_topics, **self.lda_kwargs, id2word=self.dictionary)

    def transform(self, texts, return_probas=True):
        
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        embs = np.zeros(shape=(len(texts), self.n_topics))
        for i, doc in enumerate(corpus):
            for idx, val in self.lda[doc]:
                embs[i, idx] = val if return_probas else 1

        return embs
    
    def fit_transform(self, texts, return_probas=True):

        self.dictionary = Dictionary(texts)

        corpus = [self.dictionary.doc2bow(text) for text in texts]

        self.lda = LdaModel(corpus, num_topics=self.n_topics, **self.lda_kwargs)

        embs = np.zeros(shape=(len(texts), self.n_topics))
        for i, doc in enumerate(corpus):
            for idx, val in self.lda[doc]:
                embs[i, idx] = val if return_probas else 1
    
        return embs
    
    def get_mean_topic_embandings(self, word_embendings, topn=20):
        
        topics_embeddings = []

        for i in tqdm(range(self.n_topics)):
            topic_embeddings = np.array([word_embendings.get(i, word_embendings["<unk>"]) 
                                          for i, _ in self.lda.show_topic(i, topn=topn)]).mean(axis=0)
            
            topics_embeddings.append(topic_embeddings)

        return np.array(topics_embeddings)



        

