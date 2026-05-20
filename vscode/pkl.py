import joblib

model = joblib.load("vscode/model.pkl")

print(model.__getstate__()['_sklearn_version'])