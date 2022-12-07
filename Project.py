import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

df = pd.read_csv("E:\\ML project\\dataset\\marketing_campaign.csv",sep="\t")
df.head()
df.info()

df.isnull().sum()
null_income = df.loc[df.Income.isnull()]
med = df.Income.median()
df.Income.fillna(med, inplace=True)
df.isnull().sum()
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], infer_datetime_format = True)

df['Age'] = 2022-df['Year_Birth']

df['TotalMnt'] = df.MntMeatProducts+df.MntWines+df.MntFruits
df['children'] = df.Kidhome+df.Teenhome
df['Income_to_spend'] = round(df['TotalMnt']/df['Income'],3)
df['TotalNumPurchases'] = df.NumWebPurchases + df.NumCatalogPurchases + df.NumStorePurchases
df['num_discounted'] = round(df['NumDealsPurchases']/df['TotalNumPurchases'],3)
df['web_to_total'] = round(df.NumWebPurchases/df.TotalNumPurchases,3)
df['catalog_to_total'] = round(df.NumCatalogPurchases/df.TotalNumPurchases,3)
df['Store_to_total'] = round(df.NumStorePurchases/df.TotalNumPurchases,3)
df.fillna(0, inplace=True)
df['num_cmp'] = df.AcceptedCmp1 + df.AcceptedCmp2 + df.AcceptedCmp3 + df.AcceptedCmp4 + df.AcceptedCmp5 + df.Response
df.boxplot(column = 'TotalMnt', by = 'children')
plt.show()

df.boxplot(column = 'Income')
plt.show()

df=df.loc[df.Income<df.Income.max()]
sns.histplot(df.Income)
plt.show()

sns.histplot(df.TotalMnt)
plt.show()
sns.boxplot(x=df.Education, y=df.Income)
plt.show()
df.boxplot(column = 'Income', by = 'Marital_Status')
plt.show()
df.Marital_Status.unique()
np.array(['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone','Absurd', 'YOLO'], dtype=object)
df.Marital_Status.value_counts()
df['Marital_Status']=df.Marital_Status.replace({'Alone':'Single'})
df = df.drop(df[df['Marital_Status'] == 'Absurd'].index)
df = df.drop(df[df['Marital_Status'] == 'YOLO'].index)
df['Marital_Status'] = df.Marital_Status.replace({'Married':'Partner', 'Together':'Partner'})
df['isParent'] = df.children.apply(lambda x: 1 if x > 0 else 0)
df['withPartner'] = df.Marital_Status.apply(lambda x: 1 if x == 'Partner' else 0)
df['famsize'] = df.children + df.withPartner + 1
df.boxplot(column = 'Income', by = 'Marital_Status')
plt.show()

df = df.loc[df.Age < 100]
sns.regplot(x = df.TotalMnt, y = df.TotalNumPurchases)
plt.show()
sns.barplot(y = 'TotalMnt', x = 'num_cmp', data = df)
plt.show()
fin_df=df.drop(["Year_Birth", "Z_CostContact", "Z_Revenue",'Kidhome','Teenhome'], axis=1)
df=df.drop(["Year_Birth", "Z_CostContact", "Z_Revenue",'Kidhome','Teenhome'], axis=1)
encode = preprocessing.LabelEncoder()
fin_df['Education']=encode.fit_transform(fin_df['Education'])
fin_df['Marital_Status']=encode.fit_transform(fin_df['Marital_Status'])
fin_df

np.isinf(fin_df.iloc[:]).sum()

fin_df.replace([np.inf, -np.inf], np.nan, inplace=True)
fin_df.dropna(axis = 0,inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(axis = 0,inplace=True)
np.all(np.isfinite(fin_df))
features = ['Income', 'TotalMnt']
fin_df2 = fin_df[features]
fin_df2 = (fin_df2 - fin_df2.mean(axis=0)) / fin_df2.std(axis=0)
#cluster using kmeans
kmeans = KMeans(n_clusters=4, random_state=0)
fin_df["Cluster"] = kmeans.fit_predict(fin_df2)
df['Cluster']=fin_df['Cluster']
pl = sns.countplot(x=fin_df["Cluster"])
pl.set_title("Distribution Of The Clusters")
plt.show()

pl = sns.scatterplot(data = fin_df,x=fin_df["TotalMnt"], y=fin_df["Income"],hue=fin_df["Cluster"])
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()
Personal = ["Age", "children", "famsize", "isParent", "withPartner"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=fin_df[i], y=fin_df["TotalMnt"], hue =fin_df["Cluster"], kind="kde")
    plt.show()


trend = ["web_to_total", "catalog_to_total", "Store_to_total", "NumDealsPurchases", "TotalNumPurchases"]

for i in trend:
    plt.figure()
    sns.boxenplot(y=fin_df[i], x =fin_df["Cluster"])
    plt.show()
sns.countplot(x=fin_df["num_cmp"],hue=fin_df["Cluster"])

product = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts","MntSweetProducts", "MntGoldProds"]

for i in product:
    plt.figure()
    sns.boxenplot(y=(fin_df[i]/fin_df['TotalMnt']),  x=fin_df["Cluster"])
    plt.show()
df.to_csv('df.csv')
