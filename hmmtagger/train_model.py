import file_parser as parser
from collections import Counter
from functools import partial
import math
import re
from config import ST, ET, UNKNOWN_WORD_SYMBOL

ln = math.log


"""
calculates the n-gram counts for orders 1 to n
"""


def calc_gram_counts(sentences, n=2, start_state=ST, end_state=ET, delim="_"):
    all_grams = [[] for _ in range(n)]
    for sen in sentences:
        tags = [start_state, *[tag for _, tag in sen], end_state]
        for order in range(n):
            ngrams = generate_ngrams(tags, order+1)
            ngrams_strings = list(map(lambda ngram: delim.join(ngram), ngrams))
            all_grams[order].extend(ngrams_strings)

    all_grams_counts = list(map(lambda grams: Counter(grams), all_grams))
    return all_grams_counts


"""
calculates the MLE of all n-grams (using the counts of orders n and n-1)
"""


def ngram_likelihood(ngram_counts, n_1_gram_counts, delim="_", f=math.log10):
    ngram_likelihoods = dict()
    for ngram, c in ngram_counts.items():
        n_1_gram = delim.join(re.split(delim, ngram)[:-1])
        n_1_gram_count = n_1_gram_counts[n_1_gram]
        ngram_likelihoods[ngram] = f(c/n_1_gram_count)
    return ngram_likelihoods


"""
calculates transition probs for unigrams and bigrams
"""


def calc_transition_prob(gram_counts):
    unigram_counts, bigram_counts = gram_counts[:2]
    n = sum([c for gram, c in unigram_counts.items()])
    unigram_likelihoods = map_dict(unigram_counts, lambda c: ln(c/n))
    return unigram_likelihoods, ngram_likelihood(bigram_counts, unigram_counts, f=ln)


"""
writes *.gram file
"""


def write_gram(gram_counts, gram_transion_probs, out_file):
    with open(out_file, 'w') as f:
        f.write("\data\\" + "\n")
        for i, counts in enumerate(gram_counts):
            count = sum([c for g, c in counts.items()])
            f.write("ngram " + str(i+1) + " = " + str(count) + "\n")
        f.write("\n")
        for i, probs in enumerate(gram_transion_probs):
            f.write("\\" + str(i+1) + "-grams\\\n")
            for gram, p in probs.items():
                f.write(str(p) + "\t" + gram.replace("_", "\t") + "\n")
            f.write("\n")


"""
calculates emission probs 
tag dict: dict(tag) -> [words]
words: all unique words in the corpus
tags: all unique tags in the corpus
"""


def calc_emission_prob(tag_dict, words, tags, smooth=True):
    # emission_dict = dict(map(lambda kv: (kv[0], f(kv[1])), tag_dict.items()))
    hist = partial(bucket_list, f=ln)
    voc_size = 0
    DELTA = 0.09
    if smooth:  # laplace smoothing
        counts = map_dict(tag_dict, partial(bucket_list, normalize=False))
        voc_size = sum(map_dict(counts, lambda count: sum(count.values())).values())
        hist = partial(laplace_smooth, voc_size=voc_size, delta=DELTA)
    emission_dict = map_dict(tag_dict, hist)
    if smooth:
        for tag in tags:
            emission_dict[tag][UNKNOWN_WORD_SYMBOL] = ln(DELTA / (len(tag_dict[tag]) + voc_size))
    inv_emissions = {w: dict() for w in words}
    for w in words:
        for t in tags:
            if emission_dict[t] and w in emission_dict[t]:
                e_p = emission_dict[t][w]
                inv_emissions[w][t] = e_p
    if smooth:
        inv_emissions[UNKNOWN_WORD_SYMBOL] = dict()
        for t in tags:
            if emission_dict[t] and UNKNOWN_WORD_SYMBOL in emission_dict[t]:
                e_p = emission_dict[t][UNKNOWN_WORD_SYMBOL]
                inv_emissions[UNKNOWN_WORD_SYMBOL][t] = e_p
    return inv_emissions


"""
write *.lex file
emission_dict: the return value of calc_emission_prob()
"""


def write_lex(emission_dict, out_file):
    with open(out_file, 'w') as f:
        for w, e_p in emission_dict.items():
            f.write(w)
            for tag, prob in e_p.items():
                f.write('\t' + tag + '\t' + str(prob))
            f.write('\n')


"""
trains the models using the supplied train file
outputs .lex & .gram parameter files
"""


def train(train_file, lex_file_out, gram_file_out, smooth):
    dics = parser.build_dicts(train_file)
    tag_seg_dict = dics["tag_seg"]
    seg_tag_dict = dics["seg_tag"]
    sentences = dics["sentences"]
    # calculate emission probs
    tags = tag_seg_dict.keys()
    words = seg_tag_dict.keys()
    emissions = calc_emission_prob(tag_seg_dict, words, tags, smooth=smooth)
    write_lex(emissions, lex_file_out)  # write *.lex file
    # calculate transitions
    ngrams_counts = calc_gram_counts(sentences)
    transitions = calc_transition_prob(ngrams_counts)
    write_gram(ngrams_counts, transitions, gram_file_out)  # write *.gram file


"""
-- helpers ---
"""


def generate_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def bucket_list(arr, normalize=True, f=math.log10):
    counts = Counter(arr)
    if normalize:
        n = len(arr)
        counts = map_dict(counts, lambda c: f(c/n))
    return counts


def identity(x):
    return x


def map_dict(dict_obj, func=identity):
    return dict(map(lambda kv: (kv[0], func(kv[1])), dict_obj.items()))


def laplace_smooth(arr, voc_size=0, delta=1, f=ln):
    counts = Counter(arr)
    n = len(arr)
    counts = map_dict(counts, lambda c: f((c+delta)/(n+voc_size*delta)))
    return counts


def main(smooth):
    lex_out = '../fuck.lex'
    gram_out = '../fuck.gram'
    if smooth:
        lex_out = '../fuck_smooth.lex'
        gram_out = '../fuck_smooth.gram'
    train('../input-files/heb-pos.train', lex_out, gram_out, smooth)


if __name__ == '__main__':
    main(True)
