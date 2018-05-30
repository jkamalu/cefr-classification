import re, string

class LexicalFeaturizer:

    def __init__(self):
        self.freq_dict = {}
        self.concrete_dict = {}

        self.construct()

    def read_freq_data(self):
        with open("../common_freq.csv") as f:
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
        with open("../concreteness.csv") as f:
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
        pattern = re.compile('([^\s\w]|_)+')
        sampleText = re.sub(pattern, '', sampleText)
        vocab = set([words.lower() for words in sampleText])
        return len(vocab)

    def lexicalDiversity (self, sampleText):
        return len(sampleText) / self.vocab(sampleText)

    def construct(self):
        self.read_freq_data()
        self.read_concreteness_data()

    def featurize(self, text):
        return self.lexicalDiversity(text)