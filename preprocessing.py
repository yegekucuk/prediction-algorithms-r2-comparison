import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("./dataset/odev.csv")

# - Unvan label encoding
unvanlar : list
le = LabelEncoder()
df["unvan"] = le.fit_transform(df["unvan"])

# - Creating inputs and outputs (x,y)
y = df["maas"]
x = df.drop(["maas"], axis=1)

# - Finding P values and performing backwards elimination
x_ols = sm.add_constant(x)
y_ols = y

model = sm.OLS(y_ols, x_ols)
results = model.fit()

# - Removing Calisan ID column and continue
x = x.drop(["Calisan ID"], axis=1)
x_ols = sm.add_constant(x)
model = sm.OLS(y_ols, x_ols)
results = model.fit()

# - Save the preprocessed dataset
x.to_csv("./dataset/x.csv", index=False)
y.to_csv("./dataset/y.csv", index=False)
