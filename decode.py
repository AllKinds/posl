import sys
import hmmtagger.tagger as hmm_tagger
from config import BASIC_MODEL_NAME, HMM_MODEL_NAME


def main(argv):
    if len(argv) >= 3:
        model, test_file = argv[:2]
        param_files = argv[2:]
        decode(model, test_file, param_files)


def decode(model, test_file, param_files):
    if model == '2':
        hmm_tagger.decode(test_file, *param_files[:2], HMM_MODEL_NAME)


if __name__ == "__main__":
    main(sys.argv[1:])
