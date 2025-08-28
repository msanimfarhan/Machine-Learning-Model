from urllib.request import urlretrieve
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
urlretrieve(medical_charges_url, 'medical.csv')
import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.io as pio
pio.renderers.default = "browser" 
medical_df = pd.read_csv('medical.csv')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

model = LinearRegression()
non_smoker_df = medical_df[medical_df.smoker == 'no']


def estimateCharges(age, w,b):
    return w * age + b

# w= 50
# b =100

# def rmse(target, prediction):
#     return np.sqrt(np.mean(np.square(target-prediction)))

# def try_parameters(w,b):
#     ages = non_smoker_df.age 
#     target = non_smoker_df.charges
#     predictions = estimateCharges(ages,w,b)

#     plt.plot(ages,predictions, 'r', alpha=0.9);
#     plt.scatter(ages,target,s=8,alpha=0.8);
#     plt.xlabel('Age');
#     plt.ylabel('Charges');
#     plt.legend(['Prediction' , 'Actual']);

#     loss =rmse(target,predictions)
#     print("RMSE Loss: ",loss)
#     plt.show()
    


# try_parameters(w,b)

# # targets = non_smoker_df['charges']
# # prediction = estimateCharges(non_smoker_df.age,w,b)

def rmse(target, prediction):
    return np.sqrt(np.mean(np.square(target-prediction)))
# print(rmse(targets,prediction))

# inputs = non_smoker_df[['age']]
# targets = non_smoker_df.charges
# # print('inputs.shape :', inputs.shape)
# # print('targets.shape :', targets.shape)


# model.fit(inputs, targets)
# prediction = model.predict(inputs)
# print(rmse(targets,prediction))

# inputs, target= non_smoker_df[['age','bmi','children']], non_smoker_df['charges']
# model=LinearRegression().fit(inputs,target)
# predictions= model.predict(inputs)
# print(predictions)


# print('loss :', loss)

# fig = px.scatter(medical_df, x='age', y='charges', color='smoker')
# fig.show()

# Creation of New Column
# fig = sns.barplot(data=medical_df, x='smoker', y='charges')
# plt.show()

smoker_codes = {'no':0,'yes':1}
medical_df['smoker_code']=medical_df.smoker.map(smoker_codes)

#Checking corelation 
# print(medical_df.charges.corr(medical_df.smoker_code))

# inputs, targets = medical_df[['age','bmi','children','smoker_code']], medical_df['charges']
# model = LinearRegression().fit(inputs,targets)
# predictions = model.predict(inputs)

# loss=rmse(targets,predictions)
# print('Loss :', loss)

sex_codes = {'female':0, 'male':1}
medical_df['sex_code']=medical_df.sex.map(sex_codes)

inputs, targets = medical_df[['age','bmi','children','smoker_code','sex_code']], medical_df['charges']
model = LinearRegression().fit(inputs,targets)
predictions = model.predict(inputs)

loss=rmse(targets,predictions)
print('Loss :', loss)

enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
enc.categories_











