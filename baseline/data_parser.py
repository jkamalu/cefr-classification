import sys, os
import numpy as np
import pickle as pkl

data_dir = './ICNALE_Written_Essays_2.3'
merged_plain_dir = '{}/Merged/Plain Text'.format(data_dir)

level_mapping = {
    'A2': 3,
    'B1': 2,
    'B2': 1,
}

def parse_merged_plain_v1():
    data = []
    labels = []
    # TODO: featurizer = LexicalFeaturizer()
    
    for path in sorted(os.listdir(merged_plain_dir)):
        file_name, file_ext = path.split('.')
        attributes = file_name.split('_')

        if attributes[3] in level_mapping:
            level = level_mapping[attributes[3]]
            print(level)
        else: 
            level = 0
            
        with open('{}/{}'.format(merged_plain_dir, path), 'r') as file:
            for sample in file:
                # TODO: data.append(featurizer.featurize(sample))
                data.append([0, 0, 2, 3])
                labels.append(level)
            
    return data, labels