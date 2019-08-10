import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("income.csv", header=0, delimiter=", ")

#print(df.iloc[0])
print(df["native-country"].value_counts())
df["sex-int"] = df["sex"].apply(lambda x: 0 if "Male" else 1)
df["native-country-int"] = df["native-country"].apply(lambda x: 0 if "United-States" else 1)

labels = df["income"]
data = df[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "native-country-int"]]


train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

classifier = RandomForestClassifier(random_state=1)
classifier.fit(train_data, train_labels)

print(classifier.feature_importances_)

score = classifier.score(test_data, test_labels)
print(score)
