# -*- coding: utf-8 -*-

import pandas as pd
import re as regex
from konlpy.tag import Kkma
#from string import punctuation


file_name = ['test_total']

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
        category = list(q.category)

        
        result = save(f, question, category)

        result.to_csv("data/test_total_pos.csv", index=False, encoding='utf-8-sig')
        print(f)


# -Xms512M -Xmx1024M 