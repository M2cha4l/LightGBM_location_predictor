import pandas as pd
import numpy as np
df = pd.DataFrame(pd.read_excel('dataset.xlsx').drop(['uid'],axis=1))
c = df.drop(['jingdu','weidu'],axis=1).columns
df.set_index(['jingdu','weidu'],drop=False,inplace=True)
print(df)
x_asxi = df['jingdu']
y_asxi = df['weidu']
for column in c:
    data_list = dict()
    for y in set(y_asxi):
        for x in x_asxi:
            # 把这个当做一个特征，每一个y作为一个feature
            value = []
            if df.loc[x,y].loc['weidu'] == y:
                value.append(df[x,y][column])
            else:
                value.append(-9999)
            data_list[y] = value
                # print(len(df.loc[x,'weidu']))
        data = pd.DataFrame(data_list).to_excel(f'./dataset/{column}.xlsx')
        print('完成',column)




