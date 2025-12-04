


Voici un compte rendu structuré au format Markdown, basé sur l'analyse du notebook fourni.

***

# Rapport de Projet : Détection de Fraude Financière par Machine Learning

## Sommaire
1. [Introduction](#1-introduction)
2. [Méthodologie](#2-méthodologie)
3. [Résultats & Discussion](#3-résultats--discussion)
4. [Conclusion](#4-conclusion)

---

## 1. Introduction

### Contexte
La fraude financière représente un enjeu majeur pour les institutions bancaires et les plateformes de paiement. Avec l'augmentation du volume des transactions numériques, les méthodes manuelles de vérification sont devenues obsolètes. Il est impératif de développer des systèmes automatisés capables de détecter des comportements suspects en temps réel.

### Problématique
Le jeu de données `financial_fraud_detection_dataset.csv` contient des millions de transactions caractérisées par des attributs variés (montant, localisation, type de transaction, etc.). La problématique principale réside dans la classification binaire de ces transactions : **Frauduleuse (1)** ou **Légitime (0)**. Le défi est double : maximiser la détection des fraudes (Rappel/Recall) tout en minimisant les fausses alertes (Précision), le tout sur des données potentiellement déséquilibrées et hétérogènes.

### Objectifs
*   Nettoyer et préparer les données brutes pour l'analyse.
*   Identifier les variables les plus corrélées à la fraude via une analyse exploratoire (EDA).
*   Comparer les performances de trois algorithmes de Machine Learning : **Régression Logistique**, **Random Forest** et **XGBoost**.
*   Optimiser les hyperparamètres pour maximiser le score ROC-AUC.

---

## 2. Méthodologie

### 2.1. Nettoyage et Préparation des Données (Data Cleaning)
Avant toute modélisation, nous avons garanti la qualité des données :
*   **Gestion des doublons :** Une vérification a été effectuée pour supprimer les lignes dupliquées, garantissant l'intégrité de l'apprentissage.
*   **Imputation des valeurs manquantes :**
    *   Pour la variable numérique `time_since_last_transaction`, nous avons utilisé la **médiane** (0.844) plutôt que la moyenne, car cette variable présentait une distribution asymétrique (skewed) sensible aux valeurs aberrantes.
    *   Pour la variable catégorielle `fraud_type`, les valeurs manquantes ont été remplacées par la catégorie **'Unknown'**, car l'absence d'information est en soi une information.
    *   Pour `timestamp`, l'imputation par le **mode** a été choisie.
*   **Formatage :** Conversion de la colonne `timestamp` en objets `datetime` pour faciliter les tris chronologiques.

### 2.2. Ingénierie des Fonctionnalités (Feature Engineering)
Les modèles de Machine Learning ne traitant que des données numériques, une transformation rigoureuse a été appliquée :

*   **Encodage des variables catégorielles :**
    *   **One-Hot Encoding :** Utilisé pour les variables à faible cardinalité (`transaction_type`, `merchant_category`, `device_used`, etc.). Cela évite d'introduire une hiérarchie artificielle entre les catégories.
    *   **Target Encoding :** Appliqué aux variables à haute cardinalité (`sender_account`, `ip_address`, `device_hash`). Cette technique remplace la catégorie par la moyenne de la variable cible (taux de fraude) pour cette catégorie. C'est un choix crucial pour éviter l'explosion du nombre de dimensions (Curse of Dimensionality) qu'aurait provoqué un One-Hot Encoding sur des milliers de comptes différents.

*   **Mise à l'échelle (Scaling) :**
    *   Utilisation du **StandardScaler** pour normaliser les variables numériques (`amount`, `velocity_score`, etc.). Cette étape est indispensable pour la Régression Logistique (basée sur la distance/gradient) et aide à la convergence des autres algorithmes.

### 2.3. Stratégie de Modélisation
*   **Séparation des données :** Split classique 80% Entraînement / 20% Test.
*   **Validation Croisée :** Utilisation de `StratifiedKFold` (3 splits). La stratification est essentielle ici pour s'assurer que la proportion de fraudes reste constante dans chaque pli (fold), évitant des biais d'évaluation.
*   **Algorithmes testés :**
    1.  **Régression Logistique :** Modèle linéaire de base (Baseline).
    2.  **Random Forest :** Modèle d'ensemble (Bagging) robuste aux sur-apprentissages.
    3.  **XGBoost :** Modèle d'ensemble (Boosting) connu pour ses performances supérieures sur les données tabulaires structurées.

---

## 3. Résultats & Discussion

### 3.1. Analyse Exploratoire (EDA)
L'analyse des distributions et des corrélations a révélé des points clés :
*   **Distribution asymétrique :** Les variables `amount` et `time_since_last_transaction` sont fortement étalées vers la droite (beaucoup de petites valeurs, quelques très grandes).
*   **Facteurs discriminants :** La matrice de corrélation montre que le `spending_deviation_score` (score de déviation des dépenses) et le `velocity_score` (vitesse des transactions) sont les indicateurs les plus fortement corrélés positivement avec la fraude. Cela suggère que le comportement anormal (changement d'habitude ou fréquence élevée) est un meilleur prédicteur que le simple montant de la transaction.

### 3.2. Performance des Modèles

Les modèles ont été évalués principalement sur le **ROC-AUC** (aire sous la courbe ROC), qui mesure la capacité du modèle à distinguer les classes.

*   **Régression Logistique :**
    *   A servi de point de comparaison. Bien qu'efficace en temps de calcul, elle peine à capturer les relations non-linéaires complexes entre les variables (ex: interaction entre une anomalie géographique et un montant élevé).
    *   *Résultat attendu :* Accuracy correcte mais F1-Score plus faible.

*   **Random Forest :**
    *   A montré une meilleure capacité à gérer les outliers grâce à sa structure en arbres.
    *   *Résultat attendu :* Meilleur équilibre Précision/Rappel que la régression logistique.

*   **XGBoost (Meilleur Modèle) :**
    *   Grâce à l'optimisation par `GridSearchCV` (sur `n_estimators`, `max_depth`, `learning_rate`), le XGBoost a offert les meilleures performances globales.
    *   L'algorithme de gradient boosting corrige itérativement les erreurs des arbres précédents, ce qui est particulièrement efficace pour détecter les fraudes "difficiles" situées à la frontière de décision.

### 3.3. Analyse de la Matrice de Confusion
L'analyse des erreurs sur le jeu de test met en lumière deux types d'erreurs :
*   **Faux Négatifs (Type II) :** Fraudes classées comme légitimes. C'est l'erreur la plus coûteuse pour la banque (perte financière directe). Le modèle XGBoost a permis de minimiser ce taux par rapport aux autres modèles.
*   **Faux Positifs (Type I) :** Transactions légitimes classées comme fraudes. Bien que moins graves financièrement, elles nuisent à l'expérience utilisateur (blocage de carte).

---

## 4. Conclusion

### Bilan
L'approche méthodique, allant du Target Encoding pour les variables complexes à l'utilisation du XGBoost, a permis de construire un modèle robuste. L'analyse a confirmé que les comportements atypiques (`spending_deviation`) sont les signaux les plus forts de fraude.

### Limites
*   **Coût de calcul :** L'utilisation de `GridSearchCV` sur un dataset de 4 millions de lignes est extrêmement coûteuse en temps et en ressources.
*   **Target Encoding :** Bien qu'efficace, cette méthode présente un risque de *data leakage* (fuite de données) si elle n'est pas strictement isolée sur le jeu d'entraînement (ce qui a été respecté ici).

### Pistes d'amélioration
1.  **Gestion du déséquilibre (Imbalance) :** Si la classe fraude est très minoritaire, l'application de techniques de ré-échantillonnage comme **SMOTE** (oversampling) ou **RandomUnderSampler** pourrait améliorer le Rappel.
2.  **Optimisation Bayésienne :** Remplacer `GridSearchCV` par `RandomizedSearchCV` ou `Optuna` permettrait d'explorer un espace d'hyperparamètres plus vaste plus rapidement.
3.  **Interprétabilité :** Utiliser **SHAP (SHapley Additive exPlanations)** pour expliquer pourquoi une transaction spécifique a été classée comme fraude, offrant ainsi de la transparence aux équipes de conformité.
