from bs4 import BeautifulSoup
import re
import numpy as np


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
"didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
"he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
"I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
"i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
"it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
"mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
"mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
"oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
"she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
"should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
"this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
"there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
"they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
"wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
"we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
"what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
"where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
"why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
"would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
"y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
"you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
"you're": "you are", "you've": "you have"}


def uniformity_and_word_bag(text):
    
    ctext = text.lower()
    ctext = BeautifulSoup(ctext, "lxml").text
    ctext = re.sub("[\(\[].*?[\)\]]", "", ctext)
    ctext = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in ctext.split(" ")])
    ctext = re.sub(r"'s\b","", ctext)
    ctext = re.sub("[^a-zA-Z]", " ", ctext)
    
    return ctext.split()


def add_summary_end(bag_text):
    
    bag_text.append('<EOS>')
    
    return bag_text


def slice_text(sentence, max_len):

    return sentence[:max_len]


def text_cleaner(sentence, is_summary, maxlen):
    
    words = uniformity_and_word_bag(sentence)
    if(is_summary):
        words = add_summary_end(words)
    words = slice_text(words, maxlen)
    
    return words


def get_vocabulary(all_sentences, all_summary):

    all_words = []

    for i in range(0, len(all_sentences)):
        all_words.extend(uniformity_and_word_bag(all_sentences[i]))

    for i in range(0, len(all_summary)):
        all_words.extend(uniformity_and_word_bag(all_summary[i]))

    all_words.append('<SOS>')
    all_words.append('<EOS>')
    all_words.append('<PAD>')
    all_words.append('<OOV>')

    vocabulary = set(all_words)

    return vocabulary


def create_embedding_vector(embedding_file, vocabulary):

    word2index = {}
    index2word = {}
    embeddings = []

    index2word[0] = '<PAD>'
    index2word[1] = '<SOS>'
    index2word[2] = '<EOS>'
    index2word[3] = '<OOV>'
    word2index['<PAD>'] = 0
    word2index['<SOS>'] = 1
    word2index['<EOS>'] = 2
    word2index['<OOV>'] = 3
    embeddings.append(np.zeros((50)))
    embeddings.append(np.random.rand(50))
    embeddings.append(np.random.rand(50))
    embeddings.append(np.random.rand(50))

    index_val = 4

    for line in embedding_file:
        values = line.split()
        word = values[0]
        if(word in vocabulary):
            vector = np.asarray(values[1:], "float32")
            word2index[word] = index_val
            index2word[index_val] = word
            embeddings.append(vector)
            index_val += 1
        else:
            continue
        
    embeddings = np.array(embeddings, dtype = np.float32)

    return embeddings, index2word, word2index
    