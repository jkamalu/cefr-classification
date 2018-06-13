import os
import re
import numpy as np
import pickle as pkl
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
import spacy

class DataParser():
    
    def __init__(self, embed_dim, max_seq_len):
        self.icnale_path = '../data/ICNALE'
        self.glove_path = '../data/glove.6B.50d.txt'
        self.embed_path = '../data/embed_matrix.pkl'
        self.seq_path = '../data/x_seq.pkl'       
        self.labels_path = '../data/labels.pkl'
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.spacy_parser = spacy.load('en_core_web_sm')
        self.labels_index = {
            'A2-0': 0,
            'B1-1': 1,
            'B1-2': 2,
            'B2-0': 3,
            'Native': 4
        }    
        
    def parse_data(self):
        raw_seq, raw_lex, raw_labels, word_index = self.build_data()
        x_seq, x_syn = self.prepare_data(raw_sequences, raw_labels)
        embed_matrix = self.prepare_embedding(word_index)        
        return sequences, labels, para_features, embed_matrix
    
    def build_data(self):
        sampler = self._sample_generator()

        texts = []

        tok_index = {}
        pos_index = {}

        tok_seqs = []
        pos_seqs = []
        
        while True:
            try:
                sample = sampler.next()
            except:
                break
            text, label = sample
            texts.append(text)

            # Build labels index
            if label[:2] == 'XX':
                label = 'Native' 
            raw_labels.append(label)
            
            # Build word index and sequences
            tok_idx_seq = []
            pos_idx_seq = []
            parse = self.spacy_parser.(text)
            for token in parse:
                if token.word not in tok_index:
                    tok_index[token.word] = len(tok_index) + 1
                if token.pos_ not in pos_index:
                    pos_index[token.pos_] = len(pos_index) + 1
                tok_idx_seq.append(tok_index[token.word])
                pos_idx_seq.append(pos_index[token.pos_])                

            tok_seqs.append(tok_idx_seq)
            pos_seqs.append(pos_idx_seq)
        
        n_grams = self._n_grams(texts, 1, 2)
        
        raw_seq = tok_seqs
        raw_syn = pos_seqs
        raw_lex = n_grams

        print('Parsed {} samples'.format(len(raw_sequences)))
        
        return raw_seq, raw_syn, raw_lex, raw_labels, word_index
    
    # sequence padding, categorical labels
    def prepare_data(self, raw_sequences, raw_labels):
        sequences = pad_sequences(raw_sequences, maxlen=self.max_seq_len, padding='post')
        labels = to_categorical([self.labels_index[l] for l in raw_labels])
        print('Shape of data: {}'.format(sequences.shape))
        print('Shape of labels: {}'.format(labels.shape))        
        return sequences, labels
    
    def shuffle_data(self, X, Y):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]        
        return X, Y
    
    def split_data(self, X, Y, split=0.8):
        num_val = int(split * X.shape[0])
        x_val = X[:-num_val]
        y_val = Y[:-num_val]
        x_train = X[-num_val:]
        y_train = Y[-num_val:]      
        return x_train, y_train, x_val, y_val
    
    def prepare_embedding(self, word_index):
        embeddings_index = {}
        with open(self.glove_path, 'rt') as fd:
            for line in fd:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        embedding_matrix = np.zeros((len(word_index) + 1, self.embed_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        with open(self.embed_path, 'w+') as fd:
            pkl.dump(embedding_matrix, fd)       
        return embedding_matrix    
    
    def data_gen(self):
        level_to_sample = {}
        for path in sorted(os.listdir(self.icnale_path)):
            file_name, file_ext = path.split('.')
            if file_name == 'W_CHN_SMK_B1_1': continue
            level = '-'.join(file_name.split('_')[3:])
            if level[:2] == 'XX':
                level = 'Native'             
            with open('{}/{}'.format(self.icnale_path, path), 'r') as fd:
                for sample in fd:
                    sample = sample.decode("utf-8-sig")
                    sample = sample.strip('\n')
                    sample = sample.strip('\r')
                    if sample == '': continue
                    
                    if level in level_to_sample:
                        level_to_sample[level].append(sample)
                    else:
                        level_to_sample[level] = [sample]
            
        samples = []
        levels = []
        for l in level_to_sample:
            print l
            samples.extend(level_to_sample[l][:400])
            levels.extend([l for i in range(400)])
        samples = np.array(samples)
        levels = np.array(levels)

        samples, levels = self.shuffle_data(samples, levels)
        print samples.shape, levels.shape

        for i in range(samples.shape[0]):
            yield samples[i], levels[i]
    
    def _n_grams(self, texts, low, high):
        n_vectorizer = CountVectorizer(ngram_range=(low, high), min_df=20)
        counts = n_vectorizer.fit_transform(texts)
        return counts.toarray().astype(int)       