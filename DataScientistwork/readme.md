<img src="photo_Elisee_DS.jpg" style="height:400px;margin-right:250px"/>

## Ilboudo Wendkouni Elisee CAC 2

## General description : About the dataset

This dataset contains 5 million synthetically generated financial transactions designed to simulate real-world behavior for fraud detection research and machine learning applications. Each transaction record includes fields such as:

Transaction Details: ID, timestamp, sender/receiver accounts, amount, type (deposit, transfer, etc.)

Behavioral Features: time since last transaction, spending deviation score, velocity score, geo-anomaly score

Metadata: location, device used, payment channel, IP address, device hash

Fraud Indicators: binary fraud label (is_fraud) and type of fraud (e.g., money laundering, account takeover)

The dataset follows realistic fraud patterns and behavioral anomalies, making it suitable for:

Binary and multiclass classification models

Fraud detection systems

Time-series anomaly detection

Feature engineering and model explainability

### Contexte
La fraude financière représente un enjeu majeur pour les institutions bancaires et les plateformes de paiement. Avec l'augmentation du volume des transactions numériques, les méthodes manuelles de vérification sont devenues obsolètes. Il est impératif de développer des systèmes automatisés capables de détecter des comportements suspects en temps réel.

### Problématique
Le jeu de données `financial_fraud_detection_dataset.csv` contient des millions de transactions caractérisées par des attributs variés (montant, localisation, type de transaction, etc.). La problématique principale réside dans la classification binaire de ces transactions : **Frauduleuse (1)** ou **Légitime (0)**. Le défi est double : maximiser la détection des fraudes (Rappel/Recall) tout en minimisant les fausses alertes (Précision), le tout sur des données potentiellement déséquilibrées et hétérogènes.

### Objectifs
*   Nettoyer et préparer les données brutes pour l'analyse.
*   Identifier les variables les plus corrélées à la fraude via une analyse exploratoire (EDA).
*   Comparer les performances de trois algorithmes de Machine Learning : **Régression Logistique**, **Random Forest** et **XGBoost**.
*   Optimiser les hyperparamètres pour maximiser le score ROC-AUC.

