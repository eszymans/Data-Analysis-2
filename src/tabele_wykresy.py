import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt


dane = pd.read_csv("data2.csv", header=None)
daneZnormalizowane = sk.preprocessing.normalize(dane, axis = 0)
print(daneZnormalizowane)

kmeans = sk.cluster.KMeans(n_clusters=3)
df = pd.DataFrame(kmeans.fit_predict(daneZnormalizowane))

kopiaDanych = dane.copy()
kopiaDanych['Cluster'] = kmeans.labels_

iteracje = []
wcss = []
for k in range (2, 11):
    kmeans = sk.cluster.KMeans(n_clusters=k)
    kmeans.fit(daneZnormalizowane)
    wcss.append(kmeans.inertia_)
    iteracje.append(kmeans.n_iter_)
    print(f'k = {k}, WCSS = {kmeans.inertia_}, Iterations = {kmeans.n_iter_}')

def wykresPunktowy(dane, tytulOsiX, tytulOsiY, ax, x, y):
    sns.scatterplot(x=dane.iloc[:, x], y=dane.iloc[:, y], hue=kopiaDanych['Cluster'], palette="deep" ,data=dane, s=300, ax=ax)
    ax.set_xlabel(tytulOsiX, size=50)
    ax.set_ylabel(tytulOsiY, size=50)
    ax.tick_params(axis='both', labelsize=30)
    ax.legend(fontsize=25, title_fontsize=50)

fig, axes = plt.subplots(3, 2, figsize = (35, 40))


wykresPunktowy(dane, "Długość działki kielkicha (cm)", "Długość płatka (cm)", axes[0][0], 0, 1)
wykresPunktowy(dane, "Długość działki kielkicha (cm)", "Szereokość działki kielkicha (cm)", axes[0][1], 0, 2)
wykresPunktowy(dane, "Długość działki kielkicha (cm)", "Szerokość płatka (cm)", axes[1][0], 0, 3)
wykresPunktowy(dane, "Długość płatka (cm)", "Szerokość działki kielicha (cm)", axes[1][1], 1, 2)
wykresPunktowy(dane, "Długość płatka (cm)", "Szerokość płatka (cm)", axes[2][0], 1, 3)
wykresPunktowy(dane, "Szerokość działki kielkicha (cm)", "Szerokość płatka (cm)", axes[2][1], 2, 3)

#fig.suptitle("Porównanie zależności między parametrami irysów (cm)", fontsize=25, y=0.02, ha="center")

plt.tight_layout(rect=[0, 0.01, 1, 1])
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), wcss, marker='o', linestyle='-', color='r', label='WCSS')
plt.title("Wartość WCSS dla kolejnych klastrów", fontsize=20)
plt.xlabel("Liczba klastrów (k)", fontsize=20)
plt.ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=20)
plt.xticks(range(2, 11), fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.show()