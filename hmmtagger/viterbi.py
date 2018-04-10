from config import ST, ET, TAGS


def backtrace_path(back_pointer, tag, word_no):
    if tag == ST:
        return []
    prev_tag = back_pointer[tag, word_no]
    if prev_tag == ST:
        return []
    return backtrace_path(back_pointer, prev_tag, word_no - 1) + [prev_tag]


def run_viterbi(tags, transition, emission, sentence):
    # viterbi = np.zeros((len(tags), len(sentence)))
    viterbi = dict()
    back_pointer = dict()

    for tag in tags:
        viterbi[tag, 0] = transition(ST, tag) + emission(sentence[0], tag)
        back_pointer[tag, 0] = ST

    for t in range(1, len(sentence)):
        for tag in tags:
            if tag == 'NN' and sentence[t] == 'ANFIM':
                my_lst = list((viterbi[tag_b, t-1] + transition(tag_b, tag) + emission(sentence[t], tag)) for tag_b in tags)
                tmp = max((viterbi[tag_b, t-1] + transition(tag_b, tag) + emission(sentence[t], tag)) for tag_b in tags)
                so = [t, sentence[t], tag, tmp]
            viterbi[tag, t] = \
                max((viterbi[tag_b, t-1] + transition(tag_b, tag) + emission(sentence[t], tag)) for tag_b in tags)
            back_pointer[tag, t] = max(tags, key=lambda tag_b: viterbi[tag_b, t-1] + transition(tag_b, tag))

    # viterbi[ET, len(sentence)] = max(viterbi[tag, len(sentence)-1] + transition(tag, ET) for tag in tags)
    back_pointer[ET, len(sentence)] = max(tags, key=lambda tag: viterbi[tag, len(sentence)-1] + transition(tag, ET))

    return backtrace_path(back_pointer, ET, len(sentence))
