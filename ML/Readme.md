# Ilboudo Wendkouni Elisee

## Sujet du projet

**Analyse de la qualité des vins portugais Vinho Verde (rouges et blancs) à partir de leurs propriétés physico-chimiques, en utilisant le machine learning**

---

## 1. Présentation du jeu de données

- **Source :** Wine Quality (UCI ML Repository)
- **Nbr d’échantillons :** 4898 vins blancs
- **Nbr de variables :** 11 variables physico-chimiques (acidité, sucres, pH, soufre, densité, alcool…) + 1 cible qualitative (“quality”)
- **Types de tâches réalisables :** classification, régression

**Variables principales :**
- fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- **Cible:** score de qualité (0 à 10)

_Illustration: Extrait du DataFrame_

| fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | ... | alcohol | quality | color |
|---------------|-----------------|-------------|---------------|-----------|-----|---------|---------|-------|
| 7.0           | 0.27            | 0.36        | 20.7          | 0.045     | ... | 8.8     | 6       | white |
| 6.3           | 0.30            | 0.34        | 1.6           | 0.049     | ... | 9.5     | 6       | white |
| ...           | ...             | ...         | ...           | ...       | ... | ...     | ...     | ...   |

---

## 2. Méthodologie détaillée

### A. Prétraitement des données

- **Vérification des valeurs manquantes :** aucune absence, toutes les colonnes “feature” sont continues
- **Transformation de la cible :**
  - *Classification binaire* : qualité ≤ 5 (0 = mauvais vin), qualité > 5 (1 = bon vin)

### B. Analyse exploratoire (EDA)

- **Distribution de la qualité des vins :**
  - La majorité des vins ont une qualité moyenne (5 ou 6)
  - Déséquilibre marqué entre les classes

_Illustration : Distribution des scores de qualité_

