import file_parser as parser
from config import BASE_PATH, EVAL_PATH, GOLD_PATH, BASIC_MODEL_NAME, HMM_MODEL_NAME
import sys


def main(argv):
    if len(argv) >= 3:
        tagged_path, gold_path, model = argv[:3]
        model_name = '?'
        if model == '1':
            model_name = BASIC_MODEL_NAME
        elif model == '2':
            model_name = HMM_MODEL_NAME
        evaluate_component(tagged_path, gold_path, model_name)


def num_of_correct_tags(tagged_s, gold_s):
    correct_tags_num = 0
    for tagged_w, gold_w in zip(tagged_s, gold_s):
        if tagged_w[1] == gold_w[1]:
            correct_tags_num += 1
    return correct_tags_num


def evaluate_component(tagged_path, gold_path, model_name, base=BASE_PATH):
    # with open('tagged_path', 'r') as tagged_f, open('gold_path', 'r') as gold_f:
    #     for line_t, line_g in zip(tagged_f, gold_f):
    tagged_sentences = parser.build_dicts(tagged_path)['sentences']
    gold_sentences = parser.build_dicts(gold_path)['sentences']
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
    eval_path = EVAL_PATH % model_name
    create_eval_file(Aj_arr, Allj_arr, A, All, eval_path, base)

    return {
        'All': All,
        'A': A,
        'Aj': Aj_arr,
        'Allj': Allj_arr,
    }


def create_eval_file(Aj, Allj, A, All, eval_path, base=BASE_PATH):
    with open(eval_path, "w+") as eval_f, open(base, "r") as base_r:
        lines = base_r.readlines()
        eval_f.writelines(lines)

        for i in range(len(Aj)):
            line = '%d %f %d \n' % (i+1, Aj[i], Allj[i])
            eval_f.writelines(line)

        eval_f.writelines('#--------------------------------------\n')
        eval_f.writelines('macro-avg %f %f' % (A, All))
    return


if __name__ == "__main__":
    main(sys.argv[1:])
