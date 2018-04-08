from config import TAGS, TAGGED_FILE
import hmmtagger.viterbi as viterbi
import file_parser as parser
from basic_tagger import GOLD_PATH, TEST_PATH, TAG_PATH, evaluate_component



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


def decode(untagged_file, lex_file, gram_file, model_name):
    emission_dict = parser.parse_emission(lex_file)
    transition_dict = parser.parse_transition(gram_file)
    sentences = parser.parse_sentences(untagged_file)

    transition = create_transition_func(transition_dict, smoothing_func=lambda x, y: -float("inf"))
    emission = create_transition_func(emission_dict,
                                      smoothing_func=lambda x, y: default_unseen_word_tag(x, y, emission_dict))

    tagged_path = TAGGED_FILE % model_name

    with open(tagged_path, "w") as tagged_file:
        for sentence in sentences:
            poss = viterbi.run_viterbi(TAGS, transition, emission, sentence)
            for word, tag in zip(sentence, poss):
                # print(word, tag)
                tagged_file.writelines(word + '\t' + tag + '\n')
            tagged_file.writelines('\n')  # end of sentence
        tagged_file.writelines('\n')  # end of file


def main():
    model_name = 'sharp'
    decode(TEST_PATH, 'input-files/fuck.lex', 'input-files/fuck.gram', model_name)
    tagged_path = 'output-files/heb-pos.%s.tagged' % model_name
    evaluate_component(tagged_path, GOLD_PATH, model_name)


if __name__ == '__main__':
    main()
