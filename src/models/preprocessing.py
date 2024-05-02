import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words("russian") + ["это"]

def texts_preprocessing(texts, language="russian"):
    stop_words = stopwords.words(language) + ["это"]

    texts = [[token.lower() for token in word_tokenize(text) if len(re.sub(r'[^\w]', '', token)) > 0] 
            for text in texts]
        
    texts = [[i for i in doc if i not in stop_words] for doc in texts]

    texts = [[i for i in doc if not i.isdigit()] for doc in texts]

    return texts
