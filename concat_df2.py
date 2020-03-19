import pandas as pd
import numpy as np


def makeCategory(file_name, category):

    df = pd.read_csv("data3/" + file_name + ".csv", names=['sentences', 'category'], engine='python', encoding='utf-8-sig')
    df = df[1:]
    tmp = np.full((len(df['sentences']),1), category)
    cate_col = pd.DataFrame(tmp, columns=['category'])
    
    result = pd.concat([df['sentences'], cate_col], axis=1)

    return result


def concat():
    df1 = makeCategory('nin202002211756심계항진', '심계항진')
    df2 = makeCategory('nin202002212114두통1', '두통')
    df3 = makeCategory('nin202002212114두통2', '두통')
    df4 = makeCategory('nin202002220002어지럼증', '어지럼증')
    df5 = makeCategory('nin202002221115구취', '구취')
    df6 = makeCategory('nin202002221803손발 저림', '손발저림')
    df7 = makeCategory('nin202002221952하지 마비', '하지마비')
    df8 = makeCategory('nin202002222226요통', '요통')
    df9 = makeCategory('nin202002230030잇몸 염증', '잇몸염증')
    df10 = 
    df11 = makeCategory('nin202003131937체중감소','체중감소')
    df12 = makeCategory('nin202003132011천명','천명')
    df13 = makeCategory('nin202003132121가래','가래')
    df14 = makeCategory('nin202003132216기침','기침')
    df15 = makeCategory('nin202003132338오심','오심')
    df16 = makeCategory('nin202003140022호흡곤란','호흡곤란')
    df17 = makeCategory('nin202003140128가슴통증','가슴통증')
    df18 = makeCategory('nin202003161346소화불량','소화불량')
    df19 = makeCategory('nin202003161536고열','고열')
    df20 = makeCategory('nin202003161646방사통','방사통')
    df21 = makeCategory('nin202003161811관절통','관절통')
    df22 = makeCategory('nin202003161907소양감','소양감')
    df23 = makeCategory('nin202003162030피로감','피로감')
    df24 = makeCategory('nin202003162112시야장애','시야장애')
    df25 = makeCategory('nin202003162223다뇨','다뇨')
    df26 = makeCategory('nin202003162322다식','다식')
    df27 = makeCategory('nin202003170046복통','복통')
    df28 = makeCategory('nin202003170155설사','설사')
    df29 = makeCategory('nin202003170312복부팽만','복부팽만')
    df30 = makeCategory('nin202003170410식욕부진','식욕부진')
    df31 = makeCategory('nin202003170451구토','구토')
    df32 = makeCategory('nin202003170547연하곤란','연하곤란')
    df33 = makeCategory('nin202003170716운동장애','운동장애')
    df34 = makeCategory('nin202003170757복시','복시')
    df35 = makeCategory('nin202003170920시력감소','시력감소')
    df36 = makeCategory('nin202003171020언어장애','언어장애')
    df37 = makeCategory('nin202003171105반신마비','반신마비')

    result = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,\
        df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df20,\
            df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,\
                df31,df32,df33,df34,df35,df36,df37,df38], axis=0)

    return result

total_df = concat()
total_df = total_df.dropna() #결측값 행 제거
total_df.to_csv('data3/total_36category.csv', index=False, encoding='utf-8')