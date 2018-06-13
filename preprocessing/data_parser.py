import os
import re
import numpy as np
import sys
import traceback
import pickle as pkl
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import ipywidgets as widgets
import tensorflow as tf
from IPython import display
from dragnn.protos import spec_pb2
from dragnn.python import graph_builder
from dragnn.python import spec_builder
from dragnn.python import load_dragnn_cc_impl  # This loads the actual op definitions
from dragnn.python import render_parse_tree_graphviz
from dragnn.python import visualization
from google.protobuf import text_format
from syntaxnet import load_parser_ops  # This loads the actual op definitions
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops
from tensorflow.python.platform import tf_logging as logging

class DataParser():
    
    def __init__(self, embed_dim, max_seq_len):
        self.icnale_path = '../data/ICNALE'
        self.glove_path = '../data/glove.6B.50d.txt'
        self.embed_path = '../data/embed_matrix.pkl'
        self.seq_path = '../data/sequences.pkl'
        self.labels_path = '../data/labels.pkl'
        self.parsey_path = '../data/parsey.pkl'
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.segmenter_model = self.load_model("../data/PARSEY_EN/segmenter", "spec.textproto", "checkpoint")
        self.parser_model = self.load_model("../data/PARSEY_EN", "parser_spec.textproto", "checkpoint")
        
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
        raw_sequences, raw_labels, word_index, label_index, raw_parsey_sequences = self.build_parsey_data()
        self.validate_sequence_len(raw_sequences, raw_labels, raw_parsey_sequences)
        sequences, labels, parsey_sequences = self.prepare_data(raw_sequences, raw_labels, label_index, raw_parsey_sequences)
        embed_matrix = self.prepare_embedding(word_index)
        return sequences, labels, embed_matrix

    def validate_sequence_len(self, raw_sequences, raw_labels, raw_parsey):
        try:
            assert len(raw_sequences) == len(raw_parsey)
            assert len(raw_sequences) == len(raw_labels)
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb) # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            print('An error occurred on line {} in statement {}'.format(line, text))
            exit(1)

        for i in range(len(raw_parsey)):
            seq = raw_sequences[i]
            parsey = raw_parsey[i]
            try:
                assert len(seq) == len(parsey)
            except AssertionError:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb) # Fixed format
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]

                print('An error occurred on line {} in statement {}'.format(line, text))
                print(len(seq))
                print(len(parsey))
                print(seq)
                print(parsey)
                raise
                exit(1)
        print("Parsey and the sequences have congruent shapes")
        return True

    
    def build_data(self):
        sampler = self._sample_generator()

        raw_sequences = []
        raw_labels = []
        word_index = {}
        labels_index = {}

        while True:
            try:
                sample = sampler.next()
            except:
                break
            text, label = sample

            # Build labels index
            if label[:2] == 'XX':
                label = 'Native'
            if label not in labels_index:
                labels_index[label] = len(labels_index)     
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
        
    def prepare_data(self, raw_sequences, raw_labels, labels_index, raw_parsey):
        """
        Tasks: sequence padding, categorical labels
        """

        sequences = pad_sequences(raw_sequences, maxlen=self.max_seq_len, padding='post')
        labels = to_categorical([labels_index[l] for l in raw_labels])
        parsey_sequences = raw_parsey
        for i in range(len(parsey_sequences)):
            sample = parsey_sequences[i]
            for j in range(self.max_seq_len - len(parsey_sequences)):
                parsey_sequences.append( ["$NO-CASE$"] )
        
        with open(self.seq_path, 'w+') as fd:
            pkl.dump(sequences, fd)
        with open(self.labels_path, 'w+') as fd:
            pkl.dump(labels, fd)
            
        print('Shape of data: {}'.format(sequences.shape))
        print('Shape of labels: {}'.format(labels.shape))        
        print('Shape of parsey: {}'.format(parsey_sequences.shape))
        
        return sequences, labels, parsey_sequences
    
    def shuffle_data(self, sequences, labels):
        indices = np.arange(sequences.shape[0])
        np.random.shuffle(indices)
        sequences = sequences[indices]
        labels = labels[indices]        
        return sequences, labels
    
    def split_data(self, sequences, labels, split=0.8):
        num_validation_samples = int(split * sequences.shape[0])

        x_train = sequences[:-num_validation_samples]
        y_train = labels[:-num_validation_samples]
        x_val = sequences[-num_validation_samples:]
        y_val = labels[-num_validation_samples:]       
        
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


    """ Parsey McParseface methods start here
        These will consist of building the data into paragraphs
        These paragraphs become sentences and they are processed by parsey
        These sentences get padded into sequences
        We join these sequences back into a list representing the whole paragraph
        Save the entirey of it all into parsey.dump
    """

    def load_model(self, base_dir, master_spec_name, checkpoint_name):
        # Read the master spec
        master_spec = spec_pb2.MasterSpec()
        with open(os.path.join(base_dir, master_spec_name), "r") as f:
            text_format.Merge(f.read(), master_spec)
        spec_builder.complete_master_spec(master_spec, None, base_dir)
        logging.set_verbosity(logging.WARN)  # Turn off TensorFlow spam.

        # Initialize a graph
        graph = tf.Graph()
        with graph.as_default():
            hyperparam_config = spec_pb2.GridPoint()
            builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
            # This is the component that will annotate test sentences.
            annotator = builder.add_annotation(enable_tracing=True)
            builder.add_saver()  # "Savers" can save and load models; here, we're only going to load.

        sess = tf.Session(graph=graph)
        with graph.as_default():
            #sess.run(tf.global_variables_initializer())
            #sess.run('save/restore_all', {'save/Const:0': os.path.join(base_dir, checkpoint_name)})
            builder.saver.restore(sess, os.path.join(base_dir, checkpoint_name))

        def annotate_sentence(sentence):
            with graph.as_default():
                return sess.run([annotator['annotations'], annotator['traces']],
                                feed_dict={annotator['input_batch']: [sentence]})
        return annotate_sentence

    def annotate_text(self,text):
        sentence = sentence_pb2.Sentence(
            text=text,
            token=[sentence_pb2.Token(word=text, start=-1, end=-1)]
        )

        # preprocess
        with tf.Session(graph=tf.Graph()) as tmp_session:
            char_input = gen_parser_ops.char_token_generator([sentence.SerializeToString()])
            preprocessed = tmp_session.run(char_input)[0]
        segmented, _ = self.segmenter_model(preprocessed)

        annotations, traces = self.parser_model(segmented[0])
    #     assert len(annotations) == 1
    #     assert len(traces) == 1
        return sentence_pb2.Sentence.FromString(annotations[0]), traces[0]


    def create_sentence_parse(self, sentences):  # put stuff in a function to not pollute global scope
        counter = 0
        seq = []
        tokens = []
        for sent in sentences:
            parse_tree, trace = self.annotate_text(sent)
            for token in parse_tree.token:
                if counter > self.max_seq_len: break
                counter += 1
                tokens.append( str(token.__getattribute__("word")) )
                seq.append(str(token))
        return (seq, tokens)

    def build_parsey_data(self):
        dp = DataParser(self.embed_dim, self.max_seq_len)
        sampler = dp._sample_generator()

#             sentences_used = []
        parsey_sequences = []     #each one of the elements here is a paragraph

        raw_sequences = []
        raw_labels = []
        word_index = {}
        labels_index = {}


        while True:
            if len(parsey_sequences) > 0: break
            try:
                sample = sampler.next()
            except:
                break
            text, label = sample
    #         counter = 0
    #         print text
            sentences = dp._to_sentences(text)
            seq, tokens = self.create_sentence_parse(sentences)     #returns a sequence and the tokens used by parsey
    #         seq = pad_sequences(seq, maxlen=Config.max_seq_len, padding='$NO-CASE$')
            parsey_sequences.append(seq)

            # Non parsey data building
            # Build labels index
            if label[:2] == 'XX':
                label = 'Native'
            if label not in labels_index:
                labels_index[label] = len(labels_index)     
            raw_labels.append(label)

            # Build word index and sequences
            sequence = []
#                 tokens = self._to_tokens(text)
            for word in tokens:
                if word not in word_index:
                    word_index[word] = len(word_index) + 1
                sequence.append(word_index[word])
            raw_sequences.append(sequence)

        print('Parsed {} samples'.format(len(raw_sequences)))

        return raw_sequences, raw_labels, word_index, labels_index, parsey_sequences

#             return (parsey_sequences)

    
    """ Parsey McParseface methods end here
        This space left intentionally blank for code clarity

    """

    
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
