
<img src="photo_Elisee_DS.jpg" style="height:400px;margin-right:250px"/>

## Ilboudo Wendkouni Elisee CAC 2

## Sujet du projet

**Analyse de la qualité des vins portugais Vinho Verde (rouges et blancs) à partir de leurs propriétés physico-chimiques avec le Machine learning**

---

## 1. Présentation du jeu de données

- **Source :** Wine Quality (UCI ML Repository)
- **Nbr d’échantillons :** 4898 vins blancs
- **Nbr de variables :** 11 variables physico-chimiques (acidité, sucres, pH, soufre, densité, alcool…) + 1 cible qualitative (“quality”)
- **Type de tâche :** classification binaire


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

**import pandas as pd
df = pd.read_csv('winequality-white.csv', sep=';')
df.head()**

- **Boxplots des variables physico-chimiques**
  - Permet la visualisation des différences entre variables et la présence de valeurs atypiques

_Illustration : Boxplot comparatif des variables_
** plt.figure(figsize=(10,6))
sns.boxplot(data=df.drop("quality", axis=1), orient="v")
plt.title("Boxplot des variables physico-chimiques")
plt.xticks(rotation=45)
plt.show()**


- **Matrice de corrélation**
  - Analyse des liens entre variables explicatives

_Illustration : Heatmap de corrélation_
**plt.figure(figsize=(9,8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Corrélation entre variables")
plt.show()**

---

### A. Prétraitement des données

- **Vérification des valeurs manquantes :**
  - Code : `df.isnull().sum()`
- **Transformation binaire de la cible :**
  - Code :  
    ```
    df['good_quality'] = df['quality'].apply(lambda x: 1 if x > 5 else 0)
    ```

### C. Découpage des données

- **Stratification et “shuffle” :**
  - La stratification préserve la proportion des classes lors des partages “train/val/test”
  - Shuffle garantit la représentativité et évite les biais dus à un éventuel classement des échantillons

_Exemple de découpage :_
**from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, stratify=y)**


---

## 3. Modélisation : K-Nearest Neighbors (k-NN)

- Implémentation du classificateur k-NN
- Choix de la valeur optimale de k par validation croisée
- Calcul de l’erreur sur l’ensemble de test (prédiction sur données non vues)

_Illustration : Courbe d’erreur en fonction de k_
** k_list = np.arange(1, 37, 2)
error_val = []
error_train = []

for k in k_list:
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
error_train.append(1 - model.score(X_train, y_train))
error_val.append(1 - model.score(X_val, y_val))

plt.figure()
plt.plot(k_list, error_train, label="Erreur Training")
plt.plot(k_list, error_val, label="Erreur Validation")
plt.xlabel("Valeur de k")
plt.ylabel("Erreur de classification")
plt.title("Évolution de l'erreur en fonction de k (k-NN)")
plt.legend()
plt.show()**

### B. Analyse exploratoire (EDA)

- **Distribution de la qualité des vins**
  - **Illustration :** Distribution des scores de qualité  
    _Code utilisé :_
    ```
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure()
    sns.countplot(x=df["quality"])
    plt.title("Distribution des scores de qualité du vin")
    plt.show()
    ```
- **Boxplot des variables physico-chimiques**
  - **Illustration :** Boxplot de chaque variable  
    _Code utilisé :_
    ```
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df.drop("quality", axis=1))
    plt.title("Boxplot des variables physico-chimiques")
    plt.show()
    ```
- **Matrice de corrélation**
  - **Illustration :** Heatmap de corrélation  
    _Code utilisé :_
    ```
    plt.figure(figsize=(9,8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Corrélation entre variables")
    plt.show()
    ```

### C. Découpage des données

- **Stratification et Shuffle**
  - Découpage train / validation / test  
    _Code utilisé :_
    ```
    from sklearn.model_selection import train_test_split
    X = df.drop(["quality", "good_quality"], axis=1)
    y = df["good_quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, shuffle=True, stratify=y)
    ```

---

## 3. Modélisation : K-Nearest Neighbors (k-NN)

- **Implémentation du classificateur k-NN**
  - _Code utilisé pour l'entraînement et les tests :_
    ```
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    ```
- **Sélection de k optimal par validation**
  - _Code pour courbe validation/performance :_
    ```
    k_list = list(range(1,37,2))
    error_train, error_val = [], []
    for k in k_list:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        error_train.append(1 - model.score(X_train, y_train))
        error_val.append(1 - model.score(X_val, y_val))
    plt.plot(k_list, error_train, label="Train")
    plt.plot(k_list, error_val, label="Validation")
    plt.legend()
    plt.title("Erreur en fonction de k")
    plt.show()
    ```
- **Évaluation finale sur le test set**
  - _Code pour obtenir l’erreur de généralisation :_
    ```
    test_score = model.score(X_test, y_test)
    print("Erreur de test :", 1 - test_score)
    ```

---

## 4. Résultats et interprétation

- **Erreur test :** obtenue sur données non vues (voir section "Évaluation finale sur le test set").
- **Illustration de l’impact de k sur biais/variance**
  - Voir **courbe d’erreur en fonction de k** ci-dessus ("Sélection de k optimal par validation").

---

## 5. Discussion, limites et perspectives

- **Importance de la stratification et du shuffle** (voir code “train_test_split”)
- **Limites** : classes déséquilibrées, overlap entre classes
- **Perspectives :**
  - Cross-validation pour choisir k  
    _exemple de code :_
    ```
    from sklearn.model_selection import cross_val_score
    cross_val_score(model, X, y, cv=5)
    ```
  - Standardisation pour k-NN  
    _exemple de code :_
    ```
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ```
  - Essai d’autres modèles (Random Forest, SVM, etc.)

---

## 6. Illustrations et codes additionnels

- **Histogrammes de chaque variable**
  - _Code illustration :_
    ```
    df.hist(bins=15, figsize=(12,10))
    plt.show()
    ```
- **Visualisation distribution de “good_quality”**
  - _Code illustration :_
    ```
    sns.countplot(x=df["good_quality"])
    plt.title("Distribution binaire de la qualité")
    plt.show()
    ```

---

## Conclusion

Ce projet applique pas à pas les principes du machine learning supervisé sur des données réelles, justifiant chaque étape par des résultats graphiques générés depuis le code Python du notebook ("ML.ipynb").

---

_Compt rendu détaillé et illustré du projet "ML.ipynb" – par Ilboudo Wendkouni Elisee_
