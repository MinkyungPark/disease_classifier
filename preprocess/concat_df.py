import pandas as pd
import numpy as np


def makeCategory(file_name, category):

    df = pd.read_csv("test_data/" + file_name + ".csv", encoding='utf-8')
    df = df[1:52]
    tmp = np.full((len(df['Question']),1), category)
    cate_col = pd.DataFrame(tmp, columns=['category'])
    
    result = pd.concat([df['Question'], cate_col], axis=1)

    return result


def concat():
    df1 = makeCategory('nin202003311541가슴통증', '가슴통증')
    df2 = makeCategory('nin202003311543고열', '고열')
    df4 = makeCategory('nin202003311545관절통', '관절통')
    df5 = makeCategory('nin202003311547구취', '구취')
    df6 = makeCategory('nin202003311550구토', '구토')
    df7 = makeCategory('nin202003311552기침', '기침')
    df8 = makeCategory('nin202003311553다뇨', '다뇨')
    df9 = makeCategory('nin202003311554다식', '다식')
    df10 = makeCategory('nin202003311557다음', '다음')
    df11 = makeCategory('nin202003311602두통','두통')
    df12 = makeCategory('nin202003311603반신마비','반신마비')
    df13 = makeCategory('nin202003311605방사통','방사통')
    df14 = makeCategory('nin202003311608복부팽만','복부팽만')
    df15 = makeCategory('nin202003311610복시','복시')
    df16 = makeCategory('nin202003311612복통','복통')
    df17 = makeCategory('nin202003311614설사','설사')
    df18 = makeCategory('nin202003311616소양감','소양감')
    df19 = makeCategory('nin202003311619소화불량','소화불량')
    df20 = makeCategory('nin202003311621손발저림','손발저림')
    df21 = makeCategory('nin202003311623시력감소','시력감소')
    df22 = makeCategory('nin202003311624시야장애','시야장애')
    df23 = makeCategory('nin202003311626식욕부진','식욕부진')
    df24 = makeCategory('nin202003311627심계항진','심계항진')
    df25 = makeCategory('nin202003311630어지럼증','어지럼증')
    df26 = makeCategory('nin202003311631언어장애','언어장애')
    df27 = makeCategory('nin202003311632연하곤란','연하곤란')
    df28 = makeCategory('nin202003311635오심','오심')
    df29 = makeCategory('nin202003311639요통','요통')
    df30 = makeCategory('nin202003311641운동장애','운동장애')
    df31 = makeCategory('nin202003311644잇몸염증','잇몸염증')
    df32 = makeCategory('nin202003311646천명','천명')
    df33 = makeCategory('nin202003311648체중감소','체중감소')
    df34 = makeCategory('nin202003311650피로감','피로감')
    df35 = makeCategory('nin202003311653하지마비','하지마비')
    df36 = makeCategory('nin202003311655호흡곤란','호흡곤란')
    df37 = makeCategory('nin202003311728가래','가래')

    result = pd.concat([df1,df2,df4,df5,df6,df7,df8,df9,df10,\
        df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df20,\
            df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,\
                df31,df32,df33,df34,df35,df36,df37], axis=0)

    return result

total_df = concat()
total_df = total_df.dropna() #결측값 행 제거
total_df.to_csv('data/test_total.csv', index=False, encoding='utf-8')