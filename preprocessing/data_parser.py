import os
import re
import numpy as np
import pickle as pkl
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class DataParser():
    
    def __init__(self, embed_dim, max_seq_len):
        self.icnale_path = '../data/ICNALE'
        self.glove_path = '../data/glove.6B.50d.txt'
        self.embed_path = '../data/embed_matrix.pkl'
        self.seq_path = '../data/sequences.pkl'
        self.labels_path = '../data/labels.pkl'
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
    def load_data(self):
        try:
            with open(self.seq_path) as fd:
                sequences = pkl.load(fd)
            with open(self.labels_path) as fd:
                labels = pkl.load(fd)
            with open(self.embed_path) as fd:
                embed_matrix = pkl.load(fd)
        except:
            print('Unable to load data.')
        return sequences, labels, embed_matrix
        
    def parse_data(self):
        raw_sequences, raw_labels, word_index, label_index = self.build_data()
        sequences, labels = self.prepare_data(raw_sequences, raw_labels, label_index)
        embed_matrix = self.prepare_embedding(word_index)
        return sequences, labels, embed_matrix
    
    def build_data(self):
        sampler = self._sample_generator()

        raw_sequences = []
        raw_labels = []
        word_index = {}
        labels_index = {
            'A2-0': 0,
            'B1-1': 1,
            'B1-2': 2,
            'B2-0': 3,
            'Native': 4
        }

        while True:
            try:
                sample = sampler.next()
            except:
                break
            text, label = sample

            # Build labels index
            if label[:2] == 'XX':
                label = 'Native' 
            raw_labels.append(label)
            
            # Build word index and sequences
            sequence = []    
            tokens = self._to_tokens(text)
            for word in tokens:
                if word not in word_index:
                    word_index[word] = len(word_index) + 1
                sequence.append(word_index[word])
            raw_sequences.append(sequence)

        print('Parsed {} samples'.format(len(raw_sequences)))
        
        return raw_sequences, raw_labels, word_index, labels_index
        
    def prepare_data(self, raw_sequences, raw_labels, labels_index):
        """
        Tasks: sequence padding, categorical labels
        """
        sequences = pad_sequences(raw_sequences, maxlen=self.max_seq_len, padding='post')
        labels = to_categorical([labels_index[l] for l in raw_labels])
        
        with open(self.seq_path, 'w+') as fd:
            pkl.dump(sequences, fd)
        with open(self.labels_path, 'w+') as fd:
            pkl.dump(labels, fd)
            
        print('Shape of data: {}'.format(sequences.shape))
        print('Shape of labels: {}'.format(labels.shape))        
        
        return sequences, labels
    
    def shuffle_data(self, sequences, labels):
        indices = np.arange(sequences.shape[0])
        np.random.shuffle(indices)
        sequences = sequences[indices]
        labels = labels[indices]        
        return sequences, labels
    
    def split_data(self, sequences, labels, split=0.8):
        num_validation_samples = int(split * sequences.shape[0])

        x_val = sequences[:-num_validation_samples]
        y_val = labels[:-num_validation_samples]
        x_train = sequences[-num_validation_samples:]
        y_train = labels[-num_validation_samples:]      
        
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

        print('Found %s word vectors.' % len(embeddings_index))        
        
        return embedding_matrix

    def balance_labels(self, sequences, labels):

        def cat_to_str(cat_label):
            return np.where(cat_label==1)[0][0]

        def str_to_cat(str_label):
            arr = np.zeros(labels.shape[1])
            arr[str_label] = 1
            return arr

        data_by_label = {}
        for i in range(len(sequences)):
            l = cat_to_str(labels[i])
            if l not in data_by_label:
                data_by_label[l] = [sequences[i]]
            else:
                data_by_label[l].append(sequences[i])

        trimmed_sequences = []
        trimmed_labels = []
        for label in data_by_label:
            trimmed_sequences.extend(data_by_label[label][:400])
            trimmed_labels.extend([str_to_cat(label) for i in range(400)])
        trimmed_sequences = np.array(trimmed_sequences)
        trimmed_labels = np.array(trimmed_labels)
        return trimmed_sequences, trimmed_labels      
    
    def _sample_generator(self):
        for path in sorted(os.listdir(self.icnale_path)):
            file_name, file_ext = path.split('.')
            if file_name == 'W_CHN_SMK_B1_1': continue
            level = '-'.join(file_name.split('_')[3:])
            with open('{}/{}'.format(self.icnale_path, path), 'r') as fd:
                for sample in fd:
                    sample = sample.decode("utf-8-sig")
                    sample = sample.strip('\n')
                    sample = sample.strip('\r')
                    if sample == '': continue
                    yield sample, level 
                    
    def _to_sentences(self, paragraph):
        stop = re.compile(r'([\.?!])')
        sentences = []
        for split in stop.split(paragraph):
            if split == '': continue
            if stop.match(split):
                sentences[-1] = sentences[-1] + split
            else:
                sentences.append(split.strip())
        return sentences
    
    def _to_tokens(self, sentence):
        sentence = re.sub(r'([^A-z0-9\s])', r' \1', sentence)
        return text_to_word_sequence(sentence, filters='')