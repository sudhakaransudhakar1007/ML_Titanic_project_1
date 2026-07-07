import joblib
import pandas as pd
import seaborn as sns
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
# -----------------------------------
#       LOAD DATASET
#------------------------------------
df = sns.load_dataset("titanic")

df =df[["pclass","sex","age","fare","survived"]]

x = df.drop("survived",axis =1)
y = df["survived"]


numeric_features = ["age","fare"]
categorical_features = ["pclass","sex"]

#--------------------------
#      Numeric pipeline
#---------------------------


numeric_pipeline = Pipeline(
    steps = [
        ("imputer",SimpleImputer(strategy = "median")),
        ("scaler",StandardScaler())
    ]
)


# -----------------------------
#       categorical pipeline
# -----------------------------

categorical_pipeline = Pipeline(
    steps = [
        ("imputer",SimpleImputer(strategy = "most_frequent")),
        ("encoder",OrdinalEncoder())
    ]
)


# ----------------------------------
#            Combine
# ----------------------------------

preprocessor = ColumnTransformer(
    transformers = [
        ("num",numeric_pipeline,numeric_features),
        ("cat",categorical_pipeline,categorical_features)
    ]
)


# ------------------------------------
#             Final pipeline
#-------------------------------------

pipeline = Pipeline(
    steps = [
        ("preprocessor",preprocessor),
        ("model",LogisticRegression())
    ]
)

pipeline.fit(x,y)


# Create The model directory if it doesn't exit

os.makedirs("model",exist_ok =True)
joblib.dump(pipeline,"model/pipeline.pkl")
print("Pipeline saved sucessfully")
