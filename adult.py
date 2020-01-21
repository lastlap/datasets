import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score

filename_train = 'data/Adult/adult.data'

df = pd.read_csv(filename_train,header=None)

data = [df]

column_names = ["age","workclass","fnlwgt","education","education-num","marital-status"\
,"occupation","relationship","race","sex","capital-gain","captial-loss","hours-per-week"\
,"country","salary"]

df.columns = column_names

salary_map={' <=50K':1,' >50K':0}
df['salary']=df['salary'].map(salary_map).astype(int)

df['country'] = df['country'].replace(' ?',np.nan)
df['workclass'] = df['workclass'].replace(' ?',np.nan)
df['occupation'] = df['occupation'].replace(' ?',np.nan)

df.dropna(how='any',inplace=True)

for dataset in data:
    dataset.loc[dataset['country'] != ' United-States', 'country'] = 'Non-US'
    dataset.loc[dataset['country'] == ' United-States', 'country'] = 'US'

workclass_map = {' Private':0, ' Self-emp-not-inc':1,' Self-emp-inc':2,' Federal-gov':3\
,' Local-gov':4,' State-gov':5,' Without-pay':6,' Never-worked':7,' 0':0}

education_map = {' Bachelors':0,' Some-college':1,' 11th':2,' HS-grad':3,' Prof-school':4\
,' Assoc-acdm':5,' Assoc-voc':6,' 9th':7,' 7th-8th':8,' 12th':9,' Masters':10,' 1st-4th':11\
,' 10th':12,' Doctorate':13,' 5th-6th':14,' Preschool':15,' 0':0}

marital_status_map = {' Married-civ-spouse':0,' Divorced':1,' Never-married':2,' Separated':3\
,' Widowed':4,' Married-spouse-absent':5,' Married-AF-spouse':6,' 0':0}

occupation_map = {' Tech-support':0,' Craft-repair':1,' Other-service':2,' Sales':3 \
,' Exec-managerial':4,' Prof-specialty':5,' Handlers-cleaners':6,' Machine-op-inspct':7 \
,' Adm-clerical':8,' Farming-fishing':9,' Transport-moving':10,' Priv-house-serv':11 \
,' Protective-serv':12,' Armed-Forces':13,' 0':0}

relationship_map = {' Wife':0,' Own-child':1,' Husband':2,' Not-in-family':3,' Other-relative':4 \
,' Unmarried':5,' 0':0}

race_map = {' Asian-Pac-Islander':0,' Amer-Indian-Eskimo':1,' Other':2,' Black':3,' 0':0,' White':4} 

sex_map = {' Female':0,' Male':1,' 0':0} 

# native_country_map = {' United-States':0,' Cambodia':1,' England':2,' Puerto-Rico':3,' Canada':4 \
# ,' Germany':5,' Outlying-US(Guam-USVI-etc)':6,' India':7,' Japan':8,' Greece':9,' South':10 \
# ,' China':11,' Cuba':12,' Iran':13,' Honduras':14,' Philippines':15,' Italy':16,' Poland':17,' Jamaica':18 \
# ,' Vietnam':19,' Mexico':20,' Portugal':21,' Ireland':22,' France':23,' Dominican-Republic':24,' Laos':25 \
# ,' Ecuador':26,' Taiwan':27,' Haiti':28,' Columbia':29,' Hungary':30,' Guatemala':31,' Nicaragua':32 \
# ,' Scotland':33,' Thailand':34,' Yugoslavia':35,' El-Salvador':36,' Trinadad&Tobago':37,' Peru':38 \
# ,' Hong':39,' Holand-Netherlands':40,' 0':0}

df['workclass']=df['workclass'].map(workclass_map).astype(int)
df['education']=df['education'].map(education_map).astype(int)
df['marital-status']=df['marital-status'].map(marital_status_map).astype(int)
df['occupation']=df['occupation'].map(occupation_map).astype(int)
df['relationship']=df['relationship'].map(relationship_map).astype(int)
df['race']=df['race'].map(race_map).astype(int)
df['sex'] = df['sex'].map(sex_map).astype(int)
# df['country']=df['country'].map(native_country_map).astype(int)
df['country'] = df['country'].map({'US':1,'Non-US':0}).astype(int)

y_train = df['salary'].values
del df['salary']

X_train, X_test, y_train, y_test = train_test_split(df.values,y_train,test_size=0.1,random_state=0)

svm_classifier = SVC(kernel='linear',C=1)
svm_classifier.fit(X_train,y_train)
y_pred  = svm_classifier.predict(X_test)
print("accuracy:",accuracy_score(y_test,y_pred))