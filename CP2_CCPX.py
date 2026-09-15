import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dados = pd.read_csv("./Data_for_UCI_named.csv")

x = dados.drop(columns=["stab","stabf"], axis=0)

y = dados["stabf"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

dados["stabf"].unique()

dados["stabf"].value_counts()

modelo = LogisticRegression()

modelo.fit(x_train,y_train)

y_predict = modelo.predict(x_test)

acuracia = 100 * accuracy_score(y_test, y_predict)

print(acuracia)