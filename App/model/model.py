import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


# read dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                 "../data/diabetes_data_upload.csv"))

# replace 'Yes' and 'No' with 1 and 0 respectively
cols = ["Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss", "weakness", "Polyphagia", "Genital thrush", "visual blurring",
        "Itching", "Irritability", "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity", "class"]
yn_cols = cols[2:16]

for i in yn_cols:
    df[i] = df[i].map({'Yes': 1, 'No': 0})

# replace 'Male' and 'Female' with 1 and o respectively
df["Gender"] = df["Gender"].map({'Male': 1, 'Female': 0})

# replace 'Positive' and 'Negative' with 1 and 0 respectively
df["class"] = df["class"].map({'Positive': 1, 'Negative': 0})

# split dataset into train and test set
X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# normalization and balancing the class


def scale_dataset(X, y, oversample=False):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y


# scale dataset on train and test
train, X_train, y_train = scale_dataset(X_train, y_train, oversample=True)
test, X_test, y_test = scale_dataset(X_test, y_test, oversample=False)

# Based on the machine learning analysis above, the SVM had the highest accuracy, precision, and recall.
model = SVC().fit(X_train, y_train)
y_pred = model.predict(X_test)

# serialize the model
data = {"model": model, "dataframe": df, "y_test": y_test, "y_pred": y_pred}
with open('App/model/svc_model.pkl', 'wb') as file:
    pickle.dump(data, file)
