from config import TAGS, TAGGED_FILE, UNKNOWN_WORD_SYMBOL
import hmmtagger.viterbi as viterbi
import file_parser as parser
from basic_tagger import GOLD_PATH, TEST_PATH, TAG_PATH
from evaluate import evaluate_component


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


def laplace_smooth_unseen_word(word, tag, emission_dict):
    laplace_smooth_unseen_word.counter += 1
    if (UNKNOWN_WORD_SYMBOL, tag) not in emission_dict:
        return -float("inf")
    return emission_dict[UNKNOWN_WORD_SYMBOL, tag]


laplace_smooth_unseen_word.counter = 0


def decode2(sentences, lex_file, gram_file):
    emission_dict = parser.parse_emission(lex_file)
    transition_dict = parser.parse_transition(gram_file)

    transition = create_transition_func(transition_dict,
                                        smoothing_func=lambda x, y: -float("inf"))
    emission = create_transition_func(emission_dict,
                                      smoothing_func=lambda x, y:
                                      laplace_smooth_unseen_word(x, y, emission_dict))
    tagged = list(map(lambda sen: viterbi.run_viterbi(TAGS, transition, emission, sen), sentences))
    return tagged


def decode(untagged_file, lex_file, gram_file, smooth):
    emission_dict = parser.parse_emission(lex_file)
    transition_dict = parser.parse_transition(gram_file)
    sentences = parser.parse_sentences(untagged_file)

    transition = create_transition_func(transition_dict, smoothing_func=lambda x, y: -float("inf"))
    emission = create_transition_func(emission_dict,
                                      smoothing_func=lambda x, y: laplace_smooth_unseen_word(x, y, emission_dict))

    model = 'sharp'
    if smooth:
        model = 'smooth'
    tagged_path = '../heb-pos.%s.tagged' % model

    with open(tagged_path, "w") as tagged_file:
        for sentence in sentences:
            poss = viterbi.run_viterbi(TAGS, transition, emission, sentence)
            for word, tag in zip(sentence, poss):
                # print(word, tag)
                tagged_file.writelines(word + '\t' + tag + '\n')
            tagged_file.writelines('\n')  # end of sentence
        tagged_file.writelines('\n')  # end of file


def main(smooth):
    model = 'sharp'
    lex_file = '../fuck.lex'
    gram_file = '../fuck.gram'
    if smooth:
        lex_file = '../fuck_smooth.lex'
        gram_file = '../fuck_smooth.gram'
        model = 'smooth'

    decode('../input-files/heb-pos.test', lex_file, gram_file, smooth)
    tagged_path = '../heb-pos.%s.tagged' % model
    evaluate_component(tagged_path, '../input-files/heb-pos.gold', model, '../output-files/base.eval')
    print(laplace_smooth_unseen_word.counter)


if __name__ == '__main__':
    main(True)
