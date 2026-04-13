from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

data = load_iris()
model = RandomForestClassifier()
model.fit(data.data, data.target)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")