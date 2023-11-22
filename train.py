import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

memes = pd.read_csv('contour-data.csv',dtype='float32')

memes.info()
memes.shape
'''olhando as informações sobre a coluna percebe se que há valores nulos
os valores nulos das colunas solidity, e defect_height/area decorrem onde
haveria divisão por zero, a coluna solidity representa a divisão da area 
do contorno pela area do controrno + a area das areas concavas, se a area
da parte concava + a area do contorno forem 0 quer dizer que a divisão foi
de 0/0 a relação em que a area/area+area da parte concava seria de igual
para igual por isso subistituimos por 1
Já defect_height/area e defect_rel_position, não são uma relação tão
conveniente, por isso removeremos as poucas linhas que estão nulas nestes casos'''
memes.max()

memes.min()

memes['solidity'].fillna(1, inplace=True)
memes = memes.dropna(how='any',axis=0)
memes = memes.replace([np.inf],1)

text = pd.read_csv('text-data.csv',dtype='float32')

text.max()
text.min()

text.info()
text.shape

text['solidity'].fillna(1, inplace=True)
text = text.dropna(how='any',axis=0)
text = text.replace([np.inf],1)

text_tilted = pd.read_csv('text-tilted-data.csv',dtype='float32')

text_tilted.max()
text_tilted.min()

text_tilted.info()
text_tilted.shape

text_tilted['solidity'].fillna(1, inplace=True)
text_tilted = text_tilted.dropna(how='any',axis=0)
text_tilted = text_tilted.replace([np.inf],1)
text_tilted.info()
text_tilted.shape

memes['target']=0
text['target']=1
text_tilted['target']=1

qt_lines = memes.shape[0]
text_prop = 0.5
text_tilted_prop = 1 - text_prop
text = text.sample(round(qt_lines*text_prop), random_state=1)
text_tilted = text_tilted.sample(round(qt_lines*text_tilted_prop), random_state=1)

df = pd.concat([memes,text,text_tilted],ignore_index=True)
y = df['target']
x = df.drop(['target'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

x_train.info()

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
rf_model.score(x_train, y_train)
rf_model.score(x_test, y_test)
y_pred = rf_model.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('random forest')
print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
print(f'acc: {(tp+tn)/(tp+fn+fp+tn)}')
print(f'tvp: {tp/(tp+fn)}, tfp: {fp/(fp+tn)}')

rf_model = GaussianNB()
rf_model.fit(x_train, y_train)
rf_model.score(x_train, y_train)
rf_model.score(x_test, y_test)
y_pred = rf_model.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('gaussiannb')
print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
print(f'acc: {(tp+tn)/(tp+fn+fp+tn)}')
print(f'tvp: {tp/(tp+fn)}, tfp: {fp/(fp+tn)}')
