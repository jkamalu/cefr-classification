import re, string
from sklearn.feature_extraction.text import CountVectorizer

class LexicalFeaturizer:

    def __init__(self):
        self.freq_dict = {}
        self.concrete_dict = {}

        self.construct()
        
    def n_grams(texts, low, high):
        n_vectorizer = CountVectorizer(ngram_range=(low, high))
        counts = n_vectorizer.fit_transform(texts)
        print n_vectorizer.get_feature_names()
        return counts.toarray().astype(int) 

    def read_freq_data(self):
        with open("../data/common_freq.csv") as f:
            content = f.readlines()
        content = [x.strip().split(",") for x in content[1:]]
        for word_data in content:
            word = word_data[1].strip()
            if word in self.freq_dict:
                data_obj = self.freq_dict[word]
            else:
                data_obj = {}
            data_obj["Rank"] = word_data[0]
            data_obj["PoS"] = word_data[2]
            data_obj["Frequency"] = word_data[3]
            data_obj["Dispersion"] = word_data[4]
            self.freq_dict[word] = data_obj
        return self.freq_dict

    def read_concreteness_data(self):
        with open("../data/concreteness.csv") as f:
            content = f.readlines()
        content = [x.strip().split(",") for x in content[1:]]
        for word_data in content:
            word = word_data[0].strip()
            if word in self.concrete_dict:
                data_obj = self.concrete_dict[word]
            else:
                data_obj = {}
            data_obj["Concreteness"] = word_data[2]
            data_obj["Concrete_SD"] = word_data[3]
            self.concrete_dict[word] = data_obj
        return self.concrete_dict

    def vocab (self, sampleText):
        # pattern = re.compile('([^\s\w]|_)+')
        # sampleText = re.sub(pattern, '', sampleText)
        vocab = set([words.lower() for words in sampleText])
        return len(vocab)

    def lexicalDiversity (self, sampleText):
        return len(sampleText) / self.vocab(sampleText)

    def construct(self):
        self.read_freq_data()
        self.read_concreteness_data()

    def featurize(self, text):
        # print("Featurizing now")
        features = []

        pattern = re.compile('([^\s\w]|_)+')
        unspecial_text = re.sub(pattern, '', text)
        text_list = unspecial_text.split(" ")

        # lexical diversity
        features.append(self.lexicalDiversity(text_list))

        concrete_score = 0.0
        frequency_score = 0.0
        for word in text_list:
            # print(word)
            if word in self.concrete_dict:
                concrete_score += float(self.concrete_dict[word]["Concreteness"])
            if word in self.freq_dict:
                # print(self.freq_dict[word]["Frequency"])
                frequency_score += 5000.0 - float(self.freq_dict[word]["Rank"])

        concrete_score /= (5*len(text_list))
        frequency_score /= (5000*len(text_list))
        features.append(concrete_score)
        features.append(frequency_score)
        return features