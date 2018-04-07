import re
from file_parser import build_dicts
from basic_tagger import GOLD_PATH, TEST_PATH, TAG_PATH, evaluate_component
from hmmtagger.prob_builder import read_lex
from config import ST, ET, TAGS


def backtrace_path(back_pointer, tag, word_no):
    if tag == ST:
        return []
    prev_tag = back_pointer[tag, word_no]
    if prev_tag == ST:
        return []
    return backtrace_path(back_pointer, prev_tag, word_no - 1) + [prev_tag]


def create_transition_func(transition_dict, smoothing_func):
    return lambda x, y: \
        transition_dict[x, y] if (x, y) in transition_dict else smoothing_func(x, y)


def default_unseen_word_tag(word, tag, emission_dict):
    if tag == 'NNP':
        for s in TAGS:
            if (word, s) in emission_dict:
                return -float("inf")
        # if the word is not in the dictionary, tag it as NNP
        return 0
    return -float("inf")


def run_viterbi(tags, transition_dict, emission_dict, sentence):
    # viterbi = np.zeros((len(tags), len(sentence)))
    viterbi = dict()
    back_pointer = dict()

    transition = create_transition_func(transition_dict, smoothing_func=lambda x, y: -float("inf"))
    emission = create_transition_func(emission_dict, smoothing_func=lambda x, y: default_unseen_word_tag(x, y, emission_dict))

    for tag in tags:
        viterbi[tag, 0] = transition(ST, tag) + emission(sentence[0], tag)
        back_pointer[tag, 0] = ST

    for t in range(1, len(sentence)):
        for tag in tags:
            if tag == 'yyDOT' and sentence[t] == 'yyDOT':
                my_lst = list((viterbi[tag_b, t-1] + transition(tag_b, tag) + emission(sentence[t], tag)) for tag_b in tags)
                tmp = max((viterbi[tag_b, t-1] + transition(tag_b, tag) + emission(sentence[t], tag)) for tag_b in tags)
                so = [t, sentence[t], tag, tmp]
            viterbi[tag, t] = \
                max((viterbi[tag_b, t-1] + transition(tag_b, tag) + emission(sentence[t], tag)) for tag_b in tags)
            back_pointer[tag, t] = max(tags, key=lambda tag_b: viterbi[tag_b, t-1] + transition(tag_b, tag))

    # viterbi[ET, len(sentence)] = max(viterbi[tag, len(sentence)-1] + transition(tag, ET) for tag in tags)
    back_pointer[ET, len(sentence)] = max(tags, key=lambda tag: viterbi[tag, len(sentence)-1] + transition(tag, ET))

    return backtrace_path(back_pointer, ET, len(sentence))


def parse_emission(lex_path):
    return read_lex(lex_path)


def parse_transition(gram_path):
    transition_dict = dict()
    with open(gram_path, 'r') as f:
        in_2gram_section = False
        for line in f:
            if not in_2gram_section:
                if line != '\\2-grams\\\n':
                    continue
                else:
                    in_2gram_section = True
                    continue
        # 2 Gram Section
            # Read till the empty line
            if line == '\n':
                break

            s_line = str.strip(line)
            bigram = re.split(r'\t+', s_line)
            logprob = float(bigram.pop(0))
            transition_dict[tuple(bigram)] = logprob

    return transition_dict


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


def decode(untagged_file, lex_file, gram_file, model_name):
    emission_dict = parse_emission(lex_file)
    transition_dict = parse_transition(gram_file)

    sentences = parse_sentences(untagged_file)

    tagged_path = 'output-files/heb-pos.%s.tagged' % model_name

    with open(tagged_path, "w") as tagged_file:
        for sentence in sentences:
            poss = run_viterbi(TAGS, transition_dict, emission_dict, sentence)
            for word, tag in zip(sentence, poss):
                print(word, tag)
                tagged_file.writelines(word + '\t' + tag + '\n')
            tagged_file.writelines('\n')  # end of sentence
        tagged_file.writelines('\n')  # end of file


if __name__ == '__main__':
    model_name = 'sharp'
    decode(TEST_PATH, 'fuck.lex', 'fuck.gram', model_name)
    tagged_path = 'output-files/heb-pos.%s.tagged' % model_name
    evaluate_component(tagged_path, GOLD_PATH, model_name)
