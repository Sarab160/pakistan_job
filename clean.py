import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_excel("ml_tech_jobs_dataset.xlsx")


# print(df.head())
print(df.info())

df["City"]=df["City"].fillna(df["City"].mode()[0])
df["Salary"]=df["Salary"].fillna(df["Salary"].mean())
df["Experience"] = df["Experience"].str.extract('(\d+)').astype(int)
print(df.info())

print(df["Experience"])

sns.boxplot(data=df)
plt.show()

q1=df["Salary"].quantile(0.25)
q3=df["Salary"].quantile(0.75)

iqr=q3-q1
min=q1-(1.5*iqr)
max=q3+(1.5*iqr)
print(df.shape)

df_clean = df[(df["Salary"] >= min) & (df["Salary"] <= max)]


df_clean["Company"] = df_clean["Company"].str.strip()
df_clean["Skills"] = df_clean["Skills"].str.lower()
df_clean.columns = df_clean.columns.str.lower().str.replace(" ", "_")
df_clean = df_clean[df_clean["salary"] > 1000]

df_clean.to_csv("Pakistan_job_dataset.csv",index=False)
print("done")

# df1=pd.read_csv("Pakistan_job_dataset.csv")
# sns.boxplot(data=df1)
# plt.show()