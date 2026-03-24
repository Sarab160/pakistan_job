import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

df=pd.read_csv("Pakistan_job_dataset.csv")

sns.pairplot(data=df)
# plt.show()

# print(df.head())
print(df.columns)

x=df[["experience"]]

## first encoding
en_fe=df[["city","job_type"]]
ohe=OneHotEncoder(sparse_output=False,drop="first")
encode_arr=ohe.fit_transform(en_fe)
encode_dataset=pd.DataFrame(data=encode_arr,columns=ohe.get_feature_names_out(en_fe.columns))

x1=pd.concat([x,encode_dataset],axis=1)

## second encoding

le=LabelEncoder()
en1_da=le.fit_transform(df["job_title"])
en1_dataset=pd.DataFrame(data=en1_da)

x2=pd.concat([x1,en1_dataset],axis=1)
## third encoding

cv=CountVectorizer()
en2_da=cv.fit_transform(df["skills"])
en2_dataset=pd.DataFrame(
    en2_da.toarray(),
    columns=cv.get_feature_names_out())

## final dataframe
X=pd.concat([x2,en2_dataset],axis=1)
X.columns = X.columns.astype(str)
y=df["salary"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

knr=LinearRegression()

knr.fit(x_train,y_train)
X.columns = X.columns.astype(str)

print("Test score: ",knr.score(x_test,y_test))  
print("Train score: ",knr.score(x_train,y_train))

y_pr=knr.predict(x_test)

print("Mean absolute error: ",mean_absolute_error(y_test,y_pr))
print("Mean Square error: ",mean_squared_error(y_pr,y_test))

