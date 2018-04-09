
# coding: utf-8

# In[1]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fileParser import *
from functools import reduce


# In[2]:


dicts_train = build_dicts('./input-files/heb-pos.train')
dicts_gold = build_dicts('./input-files/heb-pos.gold')

seg_tag_train = dicts_train["seg_tag"]
seg_tag_gold = dicts_gold["seg_tag"]

tag_seg_train = dicts_train["tag_seg"]
tag_seg_gold = dicts_gold["tag_seg"]


# In[3]:


def transform_to_pos_count(dictionary):
    for k,v in dictionary.items():
        dictionary[k] = len(v);

def normal_dict(dictionary,normal_sum):
    for k,v in dictionary.items():
        dictionary[k] = v/normal_sum;


# In[4]:


#transform seg_tag
def transformDicts(dict_train,dict_gold):
    transform_to_pos_count(dict_train);
    transform_to_pos_count(dict_gold);

    train_normal = sum(dict_train.values())
    gold_normal = sum(dict_gold.values())

    normal_dict(dict_train,train_normal);
    normal_dict(dict_gold,gold_normal);


# In[11]:


def plotHist(dict_train,dict_gold):
    plt.bar(list(dict_train.keys()), dict_train.values(), color='g',alpha=0.5)
    plt.bar(list(dict_gold.keys()), dict_gold.values(), color='r',alpha = 0.5)


# In[13]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#plt.rcParams["figure.figsize"] = [20,20]
transformDicts(tag_seg_train,tag_seg_gold)
plotHist(tag_seg_train,tag_seg_gold)
plt.show()


# In[ ]:

print("beforeTransform")
transformDicts(seg_tag_train,seg_tag_gold)
print("AfterTransform")
plotHist(seg_tag_train,seg_tag_gold)
print("AfterPlotHist")
plt.show()

