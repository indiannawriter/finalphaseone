
import pandas as pd
import io
import requests
import numpy as np
url='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
s=requests.get(url).content
#df=pd.read_csv(io.StringIO(s.decode('utf-8')))
df = pd.read_csv(io.StringIO(s.decode('utf-8')), na_values=['?'])
df.columns=['Scn', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'CLASS']
#this is not what they specified. they asked for SKIPNA but I solved it this way. If you want to update, that is fine.
df.A7 = pd.to_numeric(df.A7, errors='coerce')

df.to_csv("wisconsinbreast.csv")

m,n=df.shape
print(m,n)
df = df.replace('?', np.nan)
#print(df)
#print(df.mean()) 
print(df.fillna(df.mean()))

#import matplotlib.pyplot as plt
#from matplotlib.ticker import StrMethodFormatter

dfmean=df.iloc[:,[1,2,3,4,5,6,7,8,9,]].mean(axis=0)
print(dfmean)
dfmedian=df.iloc[:,[1,2,3,4,5,6,7,8,9,]].median(axis=0)
print(dfmedian)
dfstd=df.iloc[:,[1,2,3,4,5,6,7,8,9,]].std(axis=0)
print(dfstd)
dfvar=df.iloc[:,[1,2,3,4,5,6,7,8,9,]].var(axis=0)
print(dfvar)

dfscnct=df['Scn'].nunique()
print('Distinct values for Scn is', dfscnct)


#plt.hist(df)
#plt.tight_layout(rect=(0, 0, 1.2, 1.2))
#plt.xlabel('')
#plt.ylabel('')
#plt.show()


