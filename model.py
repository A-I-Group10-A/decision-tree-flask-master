import numpy as np
import pandas as pd
import pickle


data = pd.read_csv("fraudDetection.csv")

type = data["type"].value_counts()
transactions = type.index
quantity = type.values

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2,
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)


pickle.dump(model, open('model.pkl','wb'))


# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
# features = np.array([[4, 9000.60, 9000.60, 0.0]])
# print(model.predict(features))