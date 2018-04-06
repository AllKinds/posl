import file_parser as parser
from collections import Counter
from functools import partial
import math
import re

ln = math.log


def calc_gram_counts(sentences, n=2, start_state="<s>", end_state="</s>", delim="_"):
    all_grams = [[] for _ in range(n)]
    for sen in sentences:
        tags = [start_state, *[tag for _, tag in sen], end_state]
        for order in range(n):
            ngrams = generate_ngrams(tags, order+1)
            ngrams_strings = list(map(lambda ngram: delim.join(ngram), ngrams))
            all_grams[order].extend(ngrams_strings)

    all_grams_counts = list(map(lambda grams: Counter(grams), all_grams))
    return all_grams_counts


def ngram_likelihood(ngram_counts, n_1_gram_counts, delim="_", f=math.log10):
    ngram_likelihoods = dict()
    for ngram, c in ngram_counts.items():
        n_1_gram = delim.join(re.split(delim, ngram)[:-1])
        n_1_gram_count = n_1_gram_counts[n_1_gram]
        ngram_likelihoods[ngram] = f(c/n_1_gram_count)
    return ngram_likelihoods


def calc_transition_prob(gram_counts):
    unigram_counts, bigram_counts = gram_counts[:2]
    n = sum([c for gram, c in unigram_counts.items()])
    unigram_likelihoods = map_dict(unigram_counts, lambda c: ln(c/n))
    return unigram_likelihoods, ngram_likelihood(bigram_counts, unigram_counts, f=ln)


def write_gram(gram_counts, gram_transion_probs, out_file):
    with open(out_file, 'w') as f:
        f.write("\data\\" + "\n")
        for i, counts in enumerate(gram_counts):
            count = sum([c for g, c in counts.items()])
            f.write("ngram " + str(i+1) + " = " + str(count) + "\n")
        for i, probs in enumerate(gram_transion_probs):
            f.write("\\" + str(i+1) + "-grams\\\n")
            for gram, p in probs.items():
                f.write(str(p) + "\t" + gram.replace("_", "\t") + "\n")


def generate_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def calc_emission_prob2(tag_dict, ali_words, tags):
    hist = partial(bucket_list, f=ln)
    emission_dict = map_dict(tag_dict, hist)
    emission_w_t = dict()
    pass


def calc_emission_prob(tag_dict, words, all_tags):
    # emission_dict = dict(map(lambda kv: (kv[0], f(kv[1])), tag_dict.items()))
    hist = partial(bucket_list, f=ln)
    emission_dict = map_dict(tag_dict, hist)
    inv_emissions = {w: dict() for w in words}
    for w in words:
        for t in all_tags:
            e_p = 0
            if emission_dict[t] and w in emission_dict[t]:
                e_p = emission_dict[t][w]
            inv_emissions[w][t] = e_p
    return inv_emissions


def bucket_list(arr, normalize=True, f=math.log10):
    counts = Counter(arr)
    if normalize:
        n = len(arr)
        # counts = dict(map(lambda kv: (kv[0], f(kv[1]/n)), counts.items()))
        counts = map_dict(counts, lambda c: f(c/n))
    return counts


def map_dict(dict_obj, func):
    return dict(map(lambda kv: (kv[0], func(kv[1])), dict_obj.items()))


def write_lex(emission_dict, out_file):
    with open(out_file, 'w') as f:
        for w, e_p in emission_dict.items():
            f.write(w)
            for tag, prob in e_p.items():
                f.write('\t' + tag + '\t' + str(prob))
            f.write('\n')


def read_lex(path):
    emissions_dict = dict()
    with open(path) as f:
        for line in f:
            s_line = str.strip(line)
            emissions = re.split(r'\t+', s_line)
            w = emissions.pop(0)
            emissions_dict[w] = dict(zip(emissions[::2], emissions[1::2]))
    return emissions_dict


def train(train_file, lex_file, gram_file):
    dics = parser.build_dicts(train_file)
    tag_seg_dict = dics["tag_seg"]
    seg_tag_dict = dics["seg_tag"]
    sentences = dics["sentences"]
    # calculate emission probs
    all_tags = tag_seg_dict.keys()  # todo: a real list with all the possible tags
    all_words = seg_tag_dict.keys()
    emissions = calc_emission_prob(tag_seg_dict, all_words, all_tags)
    write_lex(emissions, lex_file)  # write *.lex file
    # calculate transitions
    ngrams_counts = calc_gram_counts(sentences)
    transitions = calc_transition_prob(ngrams_counts)
    write_gram(ngrams_counts, transitions, gram_file)  # write *.gram file


def test():
    train('../input-files/heb-pos.train', '../fuck.lex', '../fuck.gram')
    # dics = parser.build_dicts("../input-files/heb-pos.train")
    # emissions = calc_emission_prob(dics["tag_seg"], dics["seg_tag"].keys(), dics["tag_seg"].keys())
    # write_lex(emissions, "../fuck.lex")
    # emissions_from_file = read_lex('../fuck.lex')
    # counts = calc_gram_counts(dics["sentences"])
    # transitions = calc_transition_prob(dics["sentences"])
    # write_gram(counts, transitions, "../fuck1.gram")
    # fuck = "fuck"


if __name__ == '__main__':
    test()
