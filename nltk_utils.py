import nltk
import os
from nltk.stem.porter import PorterStemmer
import numpy as np

# Append the path to the NLTK data directory
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
stemmer = PorterStemmer()


# Return tokenized words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Return word stems in lowercase
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    all_words = [stem(word) for word in all_words]  # Stem all_words as well
    bag = np.zeros(len(all_words), dtype=np.float32)  # create a list with 0
    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0  # update to 1.0 if word exists in the sentence

    return bag

# Testing bag_of_words
# sentence = ['how', 'long', 'does', 'shipping', 'take']
# words = ['howw', 'does', 'how', 'orange']
# bag = bag_of_words(sentence, words)
# print(bag)

# Testing tokenizer
# a = 'How long does shipping take?'
# print(a)
# a = tokenize(a)
# print(a)

#
# # Testing stem
# stem_word = [stem(char) for char in a]
# print(stem_word)
