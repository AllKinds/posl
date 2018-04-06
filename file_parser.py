import re
import math
from collections import Counter


def build_dicts(path):
    # segment to POS-tag mappings
    seg_tag = dict()
    # tag to segment mappings
    tag_seg = dict()
    # sentences- 3d list
    sentences = []
    with open(path) as f:
        sentence = []
        for line in f:
            s_line = str.strip(line)
            if s_line:
                segment, tag = re.split(r'\t+', s_line)
                _map_value_to_list(seg_tag, segment, tag)
                _map_value_to_list(tag_seg, tag, segment)
                sentence.append([segment, tag])
            else:
                sentences.append(sentence)
                sentence = []

    return {
        "seg_tag": seg_tag,
        "tag_seg": tag_seg,
        "sentences": sentences
    }


def _map_value_to_list(dict_obj, key, value):
    if key in dict_obj:
        dict_obj[key].append(value)
    else:
        dict_obj[key] = [value]


"""maybe in another file"""


def get_dict_stats(dict_obj):
    unique_keys = len(dict_obj.keys())
    key_instances = sum(map(lambda l: len(l), dict_obj.values()))
    unique_values = map(lambda l: len(set(l)), dict_obj.values())
    unique_values_sum = sum(unique_values)
    print('unique:', unique_keys, 'instances:', key_instances,
          'tags per segment:', unique_values_sum / unique_keys)


""" 
    another file
    calcs P(word|tag)
"""


def calc_emission_prob(tag_dict):
    f = bucket_list
    emission_dict = dict(map(lambda kv: (kv[0], f(kv[1])), tag_dict.items()))
    return emission_dict


def bucket_list(arr, normalize=True):
    counts = Counter(arr)
    if normalize:
        n = len(arr)
        counts = dict(map(lambda kv: (kv[0], math.log2(kv[1]/n)), counts.items()))
    return counts


def main():
    path = 'input-files/heb-pos.train'
    dicts = build_dicts(path)
    get_dict_stats(dicts["seg_tag"])
    calc_emission_prob(dicts["tag_seg"])



