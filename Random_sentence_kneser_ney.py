
# coding: utf-8

# In[40]:

import pandas as pd
import collections
import matplotlib.pyplot as plt
import numpy as np
import string
import nltk
from nltk.util import ngrams
from sklearn.model_selection import train_test_split


# In[41]:

from nltk.corpus import gutenberg
from nltk.corpus import brown


# In[42]:

text_gutenberg=list(gutenberg.sents())
text_brown=list(brown.sents())
text_gutenberg_size=len(text_gutenberg)
text_brown_size=len(text_brown)
text_tr=text_brown
text_te=text_brown


# In[44]:

# Kneser-Ney implementation

words_tr=[]
for i in range(len(text_tr)):
    words_tr.extend(text_tr[i])
word_counts=len(words_tr)




bigram = ngrams(words_tr,2)
bgcounter=dict(collections.Counter(bigram))

bg_keys=list(bgcounter.keys())
total_bigram_types=len(bg_keys)



trigram = ngrams(words_tr,3)
tgcounter=dict(collections.Counter(trigram))

tg_keys=list(tgcounter.keys())
total_trigram_types=len(tg_keys)


ugcounter=dict(collections.Counter(words_tr))
ug_keys=list(ugcounter.keys())



count_first={}
for i in ug_keys:
    count_first[i]=0
for i in bg_keys:
    count_first[i[0]]=count_first[i[0]]+1
Pcont={}
norm_constant={}


count_first_bg={}
count_second_bg={}
count_second_ug={}
count_first_ug={}
count_third_ug={}
for i in tg_keys:
    count_first_bg[(i[0],i[1])]=0
    count_second_bg[(i[1],i[2])]=0
    count_second_ug[i[1]]=0
    count_first_ug[i[0]]=0
    count_third_ug[i[2]]=0
for i in tg_keys:
    count_first_bg[(i[0],i[1])]= count_first_bg[(i[0],i[1])]+1
    count_second_bg[(i[1],i[2])]= count_second_bg[(i[1],i[2])]+1
    count_second_ug[i[1]]=count_second_ug[i[1]]+1
    count_first_ug[i[0]]=count_first_ug[i[0]]+1
    count_third_ug[i[2]]=count_third_ug[i[2]]+1

for i in ug_keys:
    norm_constant[i]=(.75*(count_first[i]))/count_second_ug[i]
for i in ug_keys:
    Pcont[i]=count_third_ug[i]/total_trigram_types



def findPKn_bigram(x):
    #c=count_second_bg.get(x,0)/count_second_ug[x[0]]
    return max((count_second_bg.get(x,0)-.75),0)/count_second_ug[x[0]]+norm_constant[x[0]]*(Pcont[x[1]])


def findPKn_trigram(x):
    bgcount=findPKn_bigram((x[0],x[1]))*count_second_ug[x[0]]
    a=max((tgcounter.get(x,0)-.9),0)/bgcounter[(x[0],x[1])]
    b=(.9*count_first_bg.get((x[0],x[1]),0)*findPKn_bigram((x[1],x[2])))/bgcounter[(x[0],x[1])]
    return a+b








words_te=[]
for i in range(len(text_te)):
    words_te.extend(text_te[i])

Kneser_Ney_dict={}
for i in range(len(words_te)-2):
        x=(words_te[i],words_te[i+1],words_te[i+2])
        if(Kneser_Ney_dict.get(x,"empty")=="empty"):
            Kneser_Ney_dict[x]=findPKn_trigram(x)



# In[50]:

# random sentence generator
count=0

while count<=10:
    word=[ 'The','He','This','His','Her','Our','An'] # possible starting words
    random=np.random.choice(7,1)
    word=word[random[0]]
    sentence=[]
    for i in range(0,5):
        wordlist=[]
        for keys in tgcounter:
            if keys[0]==word:
                wordlist.append(keys[1:3])    
        if (len(wordlist)== 0):
                word='The'
                        
        for keys in tgcounter:
            if keys[0]==word:
                wordlist.append(keys[1:3])    
                
        while True:
            x=np.random.choice(len(wordlist),1)
            word_trigram=(word,wordlist[x[0]][0],wordlist[x[0]][1])
            prob =Kneser_Ney_dict[word_trigram]
            
            
            rv=np.random.random_sample()
            if rv<prob:
                sentence.append(word)
                sentence.append(word_trigram[1])
                word=word_trigram[2]
                break
    count=count+1           
        
    s=""
    a=" "
    for word in sentence:
        s=s+word+a
    print(s)
            


# In[ ]:





# In[ ]:





# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



