# Ilboudo Wendkouni Elisee

## Sujet du projet

**Analyse de la qualité des vins portugais Vinho Verde (rouges et blancs) à partir de leurs propriétés physico-chimiques avec un classificateur k-NN**

---

## 1. Présentation du jeu de données

- **Source :** Wine Quality (UCI ML Repository)
- **Nbr d’échantillons :** 4898 vins blancs
- **Nbr de variables :** 11 variables physico-chimiques (acidité, sucres, pH, soufre, densité, alcool…) + 1 cible qualitative (“quality”)
- **Type de tâche :** classification binaire

_Extrait du DataFrame : partie du code — lecture et aperçu du dataset_
import pandas as pd
df = pd.read_csv('winequality-white.csv', sep=';')
df.head()
