# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:42:44 2020

@author: Jinu
@e-mail: jwchoi@gachon.ac.kr

*reference : https://blog.naver.com/yuna1do/221691861979

Naver 지식iN 페이지에서 크롤링하여 csv 파일로 기록
2002-2019까지 연도별로 최대 1,500개 
길병원에서 온 질환, 증상 데이터에서 키워드로 검색하여 150페이지(10개씩)=1,500개씩 크롤링
제목, 질문내용, 답변내용, 날짜, URL, 검색키워드

"""

import time
import re
import pandas as pd
from datetime import datetime
import numpy as np
import requests
import lxml.html


def check_number(text):
    retext = text.replace(",","")
    lnum = int(re.findall(r'\d+', retext)[1])
    rnum = int(re.findall(r'\d+', retext)[2])
    
    if lnum == rnum:
        return True
    else:
        return False
    
        
# 전문가 답변 & 기간 설정 크롤링
def naver_crawler(quest, year):
    # 검색키워드, 검색 날짜 범위 지정 
    query = quest
    period_from =  str(year) + '.01.01.'
    period_to =  str(year) + '.12.31.'
    
    # 주소 생성 
    naver_front = 'https://kin.naver.com/search/list.nhn?query='
    period_from = '&period=' + str(period_from)
    period_to = '%7c' + str(period_to)
    kin_url = naver_front + query + period_from + period_to + '&page={}'
    
    # 크롤링 내역 담을 리스트 생성 
    title = []
    ques = []
    answer = []
    date = []
    url = []
    
    # 각 범위주소마다 페이지 범위 넘겨가며 크롤링 진행 
    page = 0
    while True:
        page += 1 
        # 점속 컨택 
        time.sleep(0.1)
        res = requests.get(kin_url.format(page)) 
        root = lxml.html.fromstring(res.text)
        
        #마지막 페이지인지 검사 후 종료
        try:
            css = root.cssselect('.number em')
            if(check_number(css[0].text_content())):
                break
        except:
            #print(check_number(css[0].text_content()))
            print("crawl error: " , kin_url.format(page))
            page -= 1
            time.sleep(60) #잦은 요청으로 에러날 경우 같음..
            next
        if page > 150:
            break
        
        
        css = root.cssselect('.basic1 dt') # 링크가 있는 루트 
        #doctor = root.cssselect('.basic1 li')
        # 각 질문마다의 링크 리스트에 담음. 
        links = []
        # 링크가 이상한 경우가 있어 https 주소로 되어있는 경우만 가져옴.
        for i in css:
            try:
                if '의사답변' in i.cssselect('img')[0].attrib['alt'] : 
                    if 'https:' in i.cssselect('a')[0].attrib['href'] :
                        links.append(i.cssselect('a')[0].attrib['href'])
                    else:
                        next
                else:
                    next
            except IndexError:
                next
        
                        
        # 각 링크마다의 제목, 답변, 게시일 담음.
        for link in links:
            #time.sleep(0.1)
            res_links = requests.get(link)
            root_links = lxml.html.fromstring(res_links.text) 

            for br in root_links.xpath("*//p"):
                br.tail = "\n" + br.tail if br.tail else "\n"
            
            for br in root_links.xpath("*//br"):
                br.tail = "\n" + br.tail if br.tail else "\n"

            try:
                title.append(root_links.cssselect('.title')[0].text_content().strip())
            except IndexError:
                title.append('')
            try:
                ques.append(root_links.cssselect('.c-heading__content')[0].text_content().strip())
            except IndexError:
                ques.append('')
            # time.sleep(0.1)
            try:
                #answer.append(re.sub(r'\s+', '',root_links.cssselect('._endContentsText')[0].text_content().strip()))
                answer.append(root_links.cssselect('._endContentsText')[0].text_content().strip())
            except IndexError:
                answer.append(np.NaN)
            # time.sleep(0.1)
            try:
                date.append(root_links.cssselect('.c-userinfo__info')[0].text_content().strip()[3:])
            except IndexError:
                date.append(np.NaN)  # 게시글이 없는 경우가 있음 결측값 처리 
            url.append(link)
        
    DF = pd.DataFrame({'Title':title, 'Question':ques,'Answer':answer,'Date':date, 'URL':url, 'Keyword':quest}) 
    return DF 

# ['체중감소','살이빠짐','몸무게가줌']
lis = [['천명','거친숨소리','쌕쌕소리'], ['가래','객담','목에가래','목에분비물','기침분비물'], \
['기침','마른기침','만성기침','잦은기침'], ['오심','구역질','속이메스껍','욕지기','토할것같'], ['호흡곤란','호흡장애','급성호흡곤란'], ['가슴통증','흉통'], \
['방사통','허리디스크','하지방사통','종아리통증'], ['관절통','퇴행성관절염','퇴행성관절증','무릎관절통','무릎관절통증','무릎부종','관절부종','관절부음'], \
['소양감','피부발진','피부가렵'],['피로감','몸이피로','몸이탈진','만성피로'], ['시야장애','시야협착','시야감소','잘안보임'], \
['다뇨','요붕증','소변자주'], ['다식','다식증','허기자주','식욕증가'], ['복통','복부통증','복부압박'], \
['설사', '과민성대장','변이무름'], ['소화불량','속이더부룩','속이불편','소화안됨'], ['복부팽만감','복부가스','복부부품','배가빵빵'], \
['열','몸에열','체온이높'], ['식욕부진','식욕없','식욕감퇴'], ['구토','토하다'], ['연하곤란','삼킴장애','연하장애'], \
['운동장애','움직임장애','손발떨림','근육경직','움직임어렵'], ['복시','사물이두개로'],['시력감소','시력저하','눈흐림','시야흐림'], \
['언어장애','발음부정확','발음이상','말더듬'],['반신마비','반신불수','편마비','']]


def run(lis):
    for li in lis:
        #연도까지 지정하고, 하나의 데이터프레임으로 합치는 함수를 만들어, 간단하게 하는 방식 구성
        df = pd.DataFrame({'Title':[0], 'Question':[0],'Answer':[0],'Date':[0], 'URL':[0], 'Keyword':[0]})

        keywords = li

        for k in keywords:
            for i in range(2009,2021): 
                df = pd.concat([df, naver_crawler(k ,i)], axis=0)        
                print(k, i, len(df))
                time.sleep(60)  #접속차단 방지
            
        #df.drop([0], inplace=True) 
        print(df) 

        filename = "C://Projects_Python//nin_data_word2vec//data//nin" + datetime.today().strftime("%Y%m%d%H%M") + keywords[0] + ".csv"
        df.to_csv(filename, encoding = 'utf-8-sig')


if __name__ == "__main__":
    run(lis)


