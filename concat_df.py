# -*- coding: utf-8 -*-

import pandas as pd
import re as regex
from konlpy.tag import Kkma
#from string import punctuation

# file_name = ['nin202002220002어지럼증', 'nin202002221115구취', 'nin202002221803손발 저림', 'nin202002221952하지 마비', 'nin202002222226요통', 'nin202002230030잇몸 염증']
# file_name = ['nin202003131937체중감소','nin202003132011천명','nin202003132121가래','nin202003132216기침',\
#     'nin202003132338오심','nin202003140022호흡곤란','nin202003140128가슴통증','nin202003161346소화불량','nin202003161536고열',\
#         'nin202003161646방사통','nin202003161811관절통','nin202003161907소양감','nin202003162030피로감',\
#             'nin202003162112시야장애','nin202003162223다뇨','nin202003162322다식','nin202003170046복통','nin202003170155설사',\
#                 'nin202003170312복부팽만','nin202003170410식욕부진','nin202003170451구토','nin202003170547연하곤란','nin202003170716운동장애',\
#                     'nin202003170757복시','nin202003170920시력감소','nin202003171020언어장애','nin202003171105반신마비']
file_name = ['다음']

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


def extract_keywords(sents):
    kkma = Kkma()
    tagged = kkma.pos(sents)

    result = []
    
    for word, tag in tagged:        
        if tag in ['NNG']: #and len(word) > 1:
            result.append(word)
        if tag in ['VV', 'VA']:
            result.append(word + "다")

    return result



def save(f, question, category):
    tmp = []
    tmp2 = []
    for q in question:
        sents = clean_text(q)
        if sents is not None:
            re_sents = extract_keywords(sents)
            re_sents = ' '.join(re_sents)
            tmp.append(re_sents)
    
    for c in category:
        tmp2.append(c)

    sents_col = pd.DataFrame(tmp, columns=['sentences'])
    cate_col = pd.DataFrame(tmp2, columns=['category'])
    result = pd.concat([sents_col, cate_col], axis=1)

    return result



if __name__ == "__main__":
    
    for f in file_name:
        q = pd.read_csv("data/" + f + ".csv", engine='python', encoding='utf-8-sig')
        q = q.dropna() #결측값 행 제거
        q.head()

        question = list(q.Question)
        category = list(q.Keyword)
        question = question[1:]
        category = category[1:]
        
        result = save(f, question, category)

        result.to_csv("data3/" + f + ".csv", index=False, encoding='utf-8-sig')
        print(f)


# -Xms512M -Xmx1024M 