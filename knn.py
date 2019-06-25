from sklearn import neighbors
from sklearn.metrics import accuracy_score
import pandas as pd
import sys

k = int(sys.argv[1]) if sys.argv else 5

df = pd.read_csv("./heart.csv")
# print(df)
# print(df.columns)

#df = df[df["Kategori"] != "?"]

x = df[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
y = df["target"]


# training
clf = neighbors.KNeighborsClassifier(k, p=2, metric="minkowski")
clf.fit(x,y)

# mengukur akurasi
y_pred = clf.predict(x)
print("Akurasi", accuracy_score(y, y_pred))


print("Hasil prediksi", clf.predict([[57,0,1,130,236,0,0,174,0,0,1,1,2]]))

