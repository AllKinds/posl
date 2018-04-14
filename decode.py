import sys
import hmmtagger.tagger as hmm_tagger
import basic_tagger
from config import BASIC_MODEL_NAME, HMM_MODEL_NAME, TAGGED_FILE
import file_parser as parser


def main(argv):
    if len(argv) >= 3:
        model, test_file = argv[:2]
        param_files = argv[2:]
        decode(model, test_file, param_files)


def decode(model, test_file, param_files):
    model_name = '?'
    if model == '1':
        model_name = BASIC_MODEL_NAME
    elif model == '2':
        model_name = HMM_MODEL_NAME

    sentences = parser.parse_sentences(test_file)
    tagged = []
    tagged_path = TAGGED_FILE % model_name
    if model == '2':
        tagged = hmm_tagger.decode2(sentences, *param_files[:2])
    elif model == '1':
        tagged = basic_tagger.decode(sentences, *param_files)

    # write tagger results
    write_tagged(tagged_path, sentences, tagged)


"""
write tagged file
sentences - the original untagged sentences
tagged - the corresponding sentences` tags (list of tag lists) 
"""


def write_tagged(path, sentences, tagged):
    with open(path, "w") as tagged_file:
        for sen, tags in zip(sentences, tagged):
            for word, tag in zip(sen, tags):
                tagged_file.writelines(word + '\t' + tag + '\n')
            tagged_file.writelines('\n')  # end of sentence
        tagged_file.writelines('\n')  # end of file


if __name__ == "__main__":
    main(sys.argv[1:])
