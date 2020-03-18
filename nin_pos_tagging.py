# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:35:11 2020

@author: Administrator

@author: Jinu
@e-mail: jwchoi@gachon.ac.kr

*reference : https://github.com/Kyubyong/wordvectors/blob/master/build_corpus.py
*reference : https://blog.theeluwin.kr/post/146591096133/%ED%95%9C%EA%B5%AD%EC%96%B4-word2vec

POS

"""

import pandas as pd
import re as regex
from konlpy.tag import Kkma
#from string import punctuation

f = ["nin202002201654고혈압"]
# f = ['nin202003131907다음','nin202003131937체중감소','nin202003132011천명','nin202003132121가래','nin202003132216기침',\
#     'nin202003132338오심','nin202003140022호흡곤란','nin202003140128가슴통증','nin202003161346소화불량','nin202003161536고열',\
#         'nin202003161646방사통','nin202003161811관절통','nin202003161858다음','nin202003161907소양감','nin202003162030피로감',\
#             'nin202003162112시야장애','nin202003162223다뇨','nin202003162322다식','nin202003170046복통','nin202003170155설사',\
#                 'nin202003170312복부팽만','nin202003170410식욕부진','nin202003170451구토','nin202003170547연하곤란','nin202003170716운동장애',\
#                     'nin202003170757복시','nin202003170920시력감소','nin202003171020언어장애','nin202003171105반신마비']


def clean_text(text):
    # Common
    text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
    text = regex.sub("&[a-z]+;", "", text) # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
    text = regex.sub("(?s){.+?}", "", text) # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
    
    text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text) # remove bold symbols
    text = regex.sub("[']{2}", "", text) # remove italic symbols
    
    #text = regex.sub(u"[^ \r\n{Hangul}.?!]", " ", text) # Replace unacceptable characters with a space.
    
    text = regex.sub("[ ]{2,}", " ", text) # Squeeze spaces.
    
    return text

def sentence_segment(text):
    sents = regex.split("([.?!])?[\n]+|[.?!]", text)
    return sents


def extract_keywords(sent):
    tagged = kkma.pos(sent)
        
    result = []
    
    for word, tag in tagged:        
        if tag in ['NNG']: #and len(word) > 1:
            result.append(word)
        if tag in ['VV', 'VA']:
            result.append(word + "다")
    return result


def word_segment(sent):   
    return extract_keywords(sent)
    #return ["{}".format(word) for word in twitter.nouns(sent)]
    


for file_name in f:
    q = pd.read_csv("data/" + file_name + ".csv", engine='python', encoding='utf-8-sig')
    q = q.dropna() #결측값 행 제거
    q.head()
    size = int(len(q) / 10)

    kkma = Kkma()

    question = list(q.Question)
    answer = list(q.Answer)


    with open("data3/" + file_name + ".csv", 'w', encoding="utf-8") as fout:
        n = 0
        for sent in question:    
            clean = clean_text(sent)
            sents = sentence_segment(clean)      

            n+=1
                
            try:
                for sent in sents:                
                    if sent is not None:
                        words = word_segment(sent)
                        if len(words) > 3:
                            fout.write(" ".join(words) + "\n")
            except:
                continue
            
            if n % size == 0 :
                print(1 + int((n / len(q) * 100/2)), "%", end = ' ')        


        # n = 0        
        # for sent in answer:    
        #     clean = clean_text(sent)
        #     sents = sentence_segment(clean)        

        #     n+=1
            
        #     try:
        #         for sent in sents:
        #             if sent is not None:
        #                 words = word_segment(sent)
        #                 if len(words) > 3:
        #                     fout.write(" ".join(words) + "\n")
        #     except:
        #         continue
            

        #     if n % size == 0 :
        #         print(51 + int((n / len(q) * 100/2)), "%", end = ' ')           
