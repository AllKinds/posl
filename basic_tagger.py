from file_parser import build_dicts

TRAIN_PATH = 'input-files/heb-pos.train'
TEST_PATH = 'input-files/heb-pos.test'
GOLD_PATH = 'input-files/heb-pos.gold'
TAG_PATH = 'output-files/heb-pos.tagged'
EVAL_PATH = 'output-files/heb-pos.basic.eval'
BASE_PATH = 'output-files/base.eval'


def train(seg_tag):
    return dict(map(lambda seg: (seg, most_common(seg_tag[seg])), seg_tag))


def most_common(tag_lst):
    return max(set(tag_lst), key=tag_lst.count)


def training_component(path):
    dicts = build_dicts(path)
    trained = train(dicts["seg_tag"])
    return trained


def tagging_component(test_path, train_params):
    # tagged_f = open("output-files/heb-pos.tagged", "w")
    with open(test_path) as test_f, open(TAG_PATH, "w") as tagged_f:
        for line in test_f:
            line = line[:-1]  # remove the \n
            if len(line) > 0:
                if line in train_params:
                    tag = train_params[line]
                else:
                    tag = 'NNP'
                line += '\t' + tag
            line += '\n'
            tagged_f.writelines(line)
    return


def num_of_correct_tags(tagged_s, gold_s):
    correct_tags_num = 0
    for tagged_w, gold_w in zip(tagged_s, gold_s):
        if tagged_w[1] == gold_w[1]:
            correct_tags_num += 1
    return correct_tags_num


def evaluate_component(tagged_path, gold_path, model_name):
    # with open('tagged_path', 'r') as tagged_f, open('gold_path', 'r') as gold_f:
    #     for line_t, line_g in zip(tagged_f, gold_f):
    tagged_sentences = build_dicts(tagged_path)['sentences']
    gold_sentences = build_dicts(gold_path)['sentences']
    nj_arr, Aj_arr, Allj_arr = [], [], []
    for tagged_s, gold_s in zip(tagged_sentences, gold_sentences):
        nj = len(tagged_s)
        Aj = (1/nj) * num_of_correct_tags(tagged_s, gold_s)
        Allj = 1 if Aj == 1 else 0
        nj_arr.append(nj)
        Aj_arr.append(Aj)
        Allj_arr.append(Allj)
    All = sum(Allj_arr)/len(gold_sentences)
    A = sum(Aj * nj for Aj, nj in zip(Aj_arr, nj_arr)) / sum(nj_arr)
    eval_path = 'output-files/heb-pos.%s.eval' % model_name
    create_eval_file(Aj_arr, Allj_arr, A, All, eval_path)

    return {
        'All': All,
        'A': A,
        'Aj': Aj_arr,
        'Allj': Allj_arr,
    }


def create_eval_file(Aj, Allj, A, All, eval_path=EVAL_PATH):
    with open(eval_path, "w+") as eval_f, open(BASE_PATH, "r") as base_r:
        lines = base_r.readlines()
        eval_f.writelines(lines)

        for i in range(len(Aj)):
            line = '%d %f %d \n' % (i+1, Aj[i], Allj[i])
            eval_f.writelines(line)

        eval_f.writelines('#--------------------------------------\n')
        eval_f.writelines('macro-avg %f %f' % (A, All))
    return


def do_stuff():
    training_params = training_component(TRAIN_PATH)
    tagging_component(TEST_PATH, training_params)
    val = evaluate_component(TAG_PATH, GOLD_PATH, 'basic')
    print(val)


if __name__ == '__main__':
    do_stuff()

