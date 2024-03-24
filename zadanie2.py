import csv
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import numpy as np


def odczytzpliku():
    with open('duze.xyz', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x, y, z = map(float, row)
            yield x, y, z

wszystkiepunkty = list(odczytzpliku())
kmeans = KMeans(n_clusters=3)
cluster = kmeans.fit_predict(wszystkiepunkty)
fig = plt.figure()
wykres = fig.add_subplot(111, projection='3d')
podzial = ['g', 'b', 'c']
for i in range(3):
    grupa = np.array(wszystkiepunkty)[cluster == i]
    wykres.scatter(grupa[:, 0], grupa[:, 1], grupa[:, 2], c=podzial[i], label=f'grupa {i + 1}')
wykres.set_xlabel('X')
wykres.set_ylabel('Y')
wykres.set_zlabel('Z')
for i in range(3):
    grupa = np.array(wszystkiepunkty)[cluster == i]
    ransac = RANSACRegressor()
    ransac.fit(grupa[:, :2], grupa[:, 2])
    vector = np.append(ransac.estimator_.coef_, -1)
    length = np.mean(np.abs(ransac.predict(grupa[:, :2]) - grupa[:, 2]))
    print(f"grupa {i + 1}:")
    print("wektor normalny plaszczyzny:", vector)
    print("srednia odleglosc punktow od plaszczyzny:", length)
    if np.abs(vector[2]) > 0.9:
        print("plaszczyzna jest pionowa")
    elif np.abs(vector[2]) < 0.1:
        print("plaszczyzna jest pozioma")
    else:
        print("plaszczyzna skosna")
plt.legend()
plt.show()
