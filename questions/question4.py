import file_parser as parser
from hmmtagger.train_model import train
from hmmtagger.tagger import decode
from evaluate import evaluate_component
from config import TAGS


def expr_partial(train_file, test_file, n=10):
    num_of_lines = parser.get_lines_count(train_file)
    lex_file_b = 'hmm.partial.%s.lex'
    gram_file_b = 'hmm.partial.%s.gram'
    tagged_b = 'hmm.partial.%s.tagged'
    gold_file = '../input-files/heb-pos.gold'
    eval_file_b = 'hmm.partial.%s.eval'
    base_eval = '../output-files/base.eval'

    for i in range(n):
        lex_file = lex_file_b % str(i+1)
        gram_file = gram_file_b % str(i + 1)
        tagged_file = tagged_b % str(i+1)
        eval_file = eval_file_b % str(i+1)
        to_read = int(((i+1)*(1/n)) * num_of_lines)
        print('to read', i, to_read)
        train(train_file, lex_file, gram_file, smooth=True,
              lines_to_read=to_read)
        decode(test_file, lex_file, gram_file, smooth=True, tagged_out=tagged_file)
        evaluate_component(tagged_file, gold_file, 'w', base=base_eval, eval_file=eval_file)


def confusion_matrix(gold_file, tagged_file):
    confusion_mat = [[0 for t in TAGS] for t2 in TAGS]
    gold_sent = parser.build_dicts(gold_file)["sentences"]
    tagged_sent = parser.build_dicts(tagged_file)["sentences"]
    for gold_s, tagged_s in zip(gold_sent, tagged_sent):
        for gold_word_tag, tagged_word_tag in zip(gold_s, tagged_s):
            _, real_tag = gold_word_tag
            _, result_tag = tagged_word_tag
            i, j = TAGS.index(result_tag), TAGS.index(real_tag)
            confusion_mat[i][j] += 1
    return confusion_mat


def print_mat(m):
    s = [[str(e) for e in row] for row in m]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    fmt_file = ' '.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    to_file = [(fmt_file.format(*row) + '\n') for row in s]
    with open('confusion_matrix.mat', 'w') as f:
        f.writelines(to_file)

    print('\n'.join(table))
    print(*enumerate(TAGS))


def get_most_common_errors(conf_mat, n=3):
    N = len(conf_mat)
    errors = [(conf_mat[x][y], x, y) for x in range(N) for y in range(N) if x != y]
    errors.sort(reverse=True, key=lambda a: a[0])
    return list(map(lambda a: (a[0], TAGS[a[1]], TAGS[a[2]]), errors[:n]))


# def printMatrix(testMatrix):
#     s = [[str(e) for e in row] for row in testMatrix]
#     print(' '),
#     for i in range(len(s[1])):  # Make it work with non square matrices.
#         print(i),
#     print()
#     for i, element in enumerate(testMatrix):
#         print()
#         i, ' '.join(element)


def main():
    # conf = confusion_matrix('../input-files/heb-pos.gold', '../heb-pos.hmm.tagged')
    # print_mat(conf)
    # err = get_most_common_errors(conf)
    # print(err)
    expr_partial('../input-files/heb-pos.train', '../input-files/heb-pos.test')


if __name__ == '__main__':
    main()


