import re
import math
from collections import Counter


"""
builds dictionaries from tagged sentences file
"""


def build_dicts(path, lines_to_read=-1):
    loop_counter = 0
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
                if lines_to_read != -1 and loop_counter >= lines_to_read:
                    break
            loop_counter += 1

    return {
        "seg_tag": seg_tag,
        "tag_seg": tag_seg,
        "sentences": sentences
    }


"""
gets sentences list from untagged file
"""


def parse_sentences(path):
    sentences = []
    with open(path) as f:
        sentence = []
        for line in f:
            s_line = str.strip(line)
            if s_line:
                segment = s_line
                sentence.append(segment)
            else:
                sentences.append(sentence)
                sentence = []

    return sentences


"""
returns bigram transitions dictionary from .gram file 
"""


def get_smoothed_transition(uni_dict, bigram_dict, delta=1):
    trans_dict = dict()
    for tag_a in uni_dict.keys():
        for tag_b in uni_dict.keys():
            trans_dict[(tag_a, tag_b)] = delta * bigram_dict[tag_a, tag_b] if (tag_a, tag_b) in bigram_dict else -float('inf')
            trans_dict[(tag_a, tag_b)] += (1 - delta) * uni_dict[tag_b]
    return trans_dict


def parse_transition(gram_path):
    bigram_dict = dict()
    unigram_dict = dict()
    with open(gram_path, 'r') as f:
        section = 0
        for line in f:
            if section == 0:
                if line == '\\1-grams\\\n':
                    section += 1
                continue
            elif section == 1:
                if line == '\n':
                    section += 1
                    continue
                s_line = str.strip(line)
                unigram = re.split(r'\t+', s_line)
                unigram_dict[unigram[1]] = float(unigram[0])

            elif section == 2:
                if line == '\\2-grams\\\n':
                    continue
                else:
                    if line == '\n':
                        section += 1
                        break

                    s_line = str.strip(line)
                    bigram = re.split(r'\t+', s_line)
                    logprob = float(bigram.pop(0))
                    bigram_dict[tuple(bigram)] = logprob

    # return bigram_dict
    return get_smoothed_transition(unigram_dict, bigram_dict)


"""
returns emission dictionary from .lex file
"""


def parse_emission(lex_path):
    emissions_dict = dict()
    with open(lex_path) as f:
        for line in f:
            s_line = str.strip(line)
            emissions = re.split(r'\t+', s_line)
            w = emissions.pop(0)
            for tag, p in zip(emissions[::2], emissions[1::2]):
                emissions_dict[(w, tag)] = float(p)
            # emissions_dict[w] = dict(zip(emissions[::2], emissions[1::2]))
    return emissions_dict


def _map_value_to_list(dict_obj, key, value):
    if key in dict_obj:
        dict_obj[key].append(value)
    else:
        dict_obj[key] = [value]


def get_word_tag_pair(path):
    w_t = dict()
    sentences = build_dicts(path)["sentences"]


def get_lines_count(file):
    with open(file) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


#
# """maybe in another file"""
#
#
# def get_dict_stats(dict_obj):
#     unique_keys = len(dict_obj.keys())
#     key_instances = sum(map(lambda l: len(l), dict_obj.values()))
#     unique_values = map(lambda l: len(set(l)), dict_obj.values())
#     unique_values_sum = sum(unique_values)
#     print('unique:', unique_keys, 'instances:', key_instances,
#           'tags per segment:', unique_values_sum / unique_keys)
#
#
# """
#     another file
#     calcs P(word|tag)
# """
#
#
# def calc_emission_prob(tag_dict):
#     f = bucket_list
#     emission_dict = dict(map(lambda kv: (kv[0], f(kv[1])), tag_dict.items()))
#     return emission_dict
#
#
# def bucket_list(arr, normalize=True):
#     counts = Counter(arr)
#     if normalize:
#         n = len(arr)
#         counts = dict(map(lambda kv: (kv[0], math.log2(kv[1]/n)), counts.items()))
#     return counts
#
#
# def main():
#     path = 'input-files/heb-pos.train'
#     dicts = build_dicts(path)
#     get_dict_stats(dicts["seg_tag"])
#     calc_emission_prob(dicts["tag_seg"])
#
#
