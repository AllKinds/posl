from file_parser import *;

from functools import reduce;


def dict_with_unique_values(dict):
    uniqe_dict = {};
    for key in dict.keys():
        values = dict[key]
        uniqe_dict[key] = []
        for value in values:
            if value not in uniqe_dict[key]:
                uniqe_dict[key].append(value)
    return uniqe_dict;


dicts_train = build_dicts('../input-files/heb-pos.train')
dicts_gold = build_dicts('../input-files/heb-pos.gold')

seg_tag_train = dicts_train["seg_tag"]
seg_tag_gold = dicts_gold["seg_tag"]



#1.1
def get_num_of_unigram_segment_instances_in_corpus(segment_tag_dictionary):
    """
    :param segment_tag_dictionary: A Dictionary containing each unique segment as a key and all references\tags as a list of values for each instance
    :return: The Number of Unigram Segment Instances
    """
    return reduce(lambda mem , key : mem+len(segment_tag_dictionary[key]) , segment_tag_dictionary.keys() , 0);

num_of_unigram_segment_instances_in_train = get_num_of_unigram_segment_instances_in_corpus(seg_tag_train)
num_of_unigram_segment_instances_in_gold = get_num_of_unigram_segment_instances_in_corpus(seg_tag_gold)
print("1.1.train: "+str(num_of_unigram_segment_instances_in_train));
print("1.1.gold: "+str(num_of_unigram_segment_instances_in_gold));

#1.2
def get_num_of_unigram_segment_kinds_in_corpus(segment_tag_dictionary):
    """
    :param segment_tag_dictionary: A Dictionary containing each unique segment as a key and all references\tags as a list of values for each instance
    :return:The Number of Unigram Segment Kinds
    """
    return len(segment_tag_dictionary.keys())

num_of_unigram_segment_kinds_in_train =get_num_of_unigram_segment_kinds_in_corpus(seg_tag_train)
num_of_unigram_segment_kinds_in_gold =get_num_of_unigram_segment_kinds_in_corpus(seg_tag_gold)
print("1.2.train: "+str(num_of_unigram_segment_kinds_in_train));
print("1.2.gold: "+str(num_of_unigram_segment_kinds_in_gold));

#1.3
def get_num_of_segment_tag_instances_in_corpus(segment_tag_dictionary):
    return get_num_of_unigram_segment_instances_in_corpus(segment_tag_dictionary)#same as 1.1?

num_of_segment_tag_instances_in_train = get_num_of_segment_tag_instances_in_corpus(seg_tag_train) #same as 1.1?
print("1.3.train: "+str(num_of_segment_tag_instances_in_train));
num_of_segment_tag_instances_in_gold = get_num_of_segment_tag_instances_in_corpus(seg_tag_gold) #same as 1.1?
print("1.3.gold: "+str(num_of_segment_tag_instances_in_gold));

#1.4
def get_num_of_segment_tags_kinds_in_corpus_from_unique(unique_segment_tag_dictionary):
    return reduce(lambda mem , key : mem+len(unique_segment_tag_dictionary[key]) , unique_segment_tag_dictionary.keys() , 0);

def get_num_of_segment_tags_kinds_in_corpus(segment_tag_dictionary):
    unique_segment_tag_dictionary =  dict_with_unique_values(segment_tag_dictionary)
    return reduce(lambda mem , key : mem+len(unique_segment_tag_dictionary[key]) , unique_segment_tag_dictionary.keys() , 0);

num_of_segment_tags_kinds_in_train = get_num_of_segment_tags_kinds_in_corpus(seg_tag_train)
print("1.4.train: "+str(num_of_segment_tags_kinds_in_train));
num_of_segment_tags_kinds_in_gold = get_num_of_segment_tags_kinds_in_corpus(seg_tag_gold)
print("1.4.gold: "+str(num_of_segment_tags_kinds_in_gold));

#1.5
def get_num_of_tags_per_segment_from_unique(unique_segment_tag_dictionary):
    num_of_tags_per_segment = list(map(lambda key:len(unique_segment_tag_dictionary[key]),unique_segment_tag_dictionary))
    return sum(num_of_tags_per_segment)/len(num_of_tags_per_segment) #ambiguity

def get_num_of_tags_per_segment(segment_tag_dictionary):
    unique_segment_tag_dictionary = dict_with_unique_values(segment_tag_dictionary)
    num_of_tags_per_segment = list(map(lambda key:len(unique_segment_tag_dictionary[key]),unique_segment_tag_dictionary))
    return sum(num_of_tags_per_segment)/len(num_of_tags_per_segment) #ambiguity


ambiguity_train = get_num_of_tags_per_segment(seg_tag_train)
print("1.5.train: "+str(ambiguity_train))
ambiguity_gold = get_num_of_tags_per_segment(seg_tag_gold)
print("1.5.gold: "+str(ambiguity_gold))




#better printing
print('////////////////////////////////////////////////')
print("better print:")
funcArray = [
    get_num_of_unigram_segment_instances_in_corpus,
    get_num_of_unigram_segment_kinds_in_corpus,
    get_num_of_segment_tag_instances_in_corpus,
    get_num_of_segment_tags_kinds_in_corpus,
    get_num_of_tags_per_segment,
]
def callFunc(function,input):
    return function(input)

def printStatistics(segment_tag_dictionary):
    funcArray = [
    get_num_of_unigram_segment_instances_in_corpus,
    get_num_of_unigram_segment_kinds_in_corpus,
    get_num_of_segment_tag_instances_in_corpus,
    get_num_of_segment_tags_kinds_in_corpus,
    get_num_of_tags_per_segment,
    ]
    for index,func in enumerate(funcArray):
        print("1."+str(index+1)+": "+str(callFunc(func,segment_tag_dictionary)))
#train
print("train statistics:")
printStatistics(seg_tag_train)

#gold
print("gold statistics:")
printStatistics(seg_tag_gold)

#1.6