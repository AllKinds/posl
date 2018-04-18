import sys
import hmmtagger.train_model as hmm_tagger
import basic_tagger
from config import GRAM_FILE, LEX_FILE, BASIC_RES_FILE


def main(argv):
    if len(argv) >= 2:
        model, train_file = argv[:2]
        smoothing = False
        if len(argv) >= 3 and argv[2] == 'y':
            smoothing = True
        train(model, train_file, smooth=smoothing)


def train(model, train_file, smooth=True):
    if model == '2':
        hmm_tagger.train(train_file, LEX_FILE, GRAM_FILE, smooth=smooth)
    elif model == '1':
        basic_tagger.train(train_file, BASIC_RES_FILE)


if __name__ == "__main__":
    main(sys.argv[1:])
