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

## dataset description

Ce dataset correspond à un problème de classification supervisée, principalement de classification binaire (fraude vs non-fraude), avec une possibilité naturelle d’étendre à une classification multi-classe si l’on exploite les types de fraude disponibles. Il peut aussi servir de base à des tâches dérivées comme la détection d’anomalies non supervisée ou la modélisation temporelle de comportements de comptes bancaires.
Problématique de modélisation
# •	Tâche principale : prédire si une transaction donnée est frauduleuse (variable cible binaire, par exemple is_fraud = 0/1).
# •	Tâche secondaire possible : prédire le type de fraude (money laundering, account takeover, etc.), ce qui devient alors une classification multi-classe sur les seules transactions frauduleuses ou sur l’ensemble des transactions.
# •	Autres usages :
# •	Détection d’anomalies / outliers sur les scores comportementaux (velocity, geo-anomaly) et les montants, sans utiliser le label pour l’apprentissage.
# •	Modèles séquentiels par compte (séries temporelles) pour détecter des ruptures de comportement (avant/après fraude).
Dans un contexte de cabinet d’études, la formulation business typique est : “à partir des informations disponibles au moment d’une transaction, est-il probable qu’elle soit frauduleuse, et de quel type de fraude s’agit-il ?”
## Caractéristiques globales du dataset
# •	Taille : environ 5 000 000 lignes, chaque ligne représentant une transaction financière individuelle.
# •	Période temporelle : transactions simulées sur une année glissante récente (2023–2024), avec une granularité fine et une répartition régulière dans le temps, ce qui est adapté à l’analyse de saisonnalité et de dérive de concept.
# •	Structure : un fichier tabulaire principal (format CSV sur Kaggle) contenant à la fois des variables d’identification, des attributs transactionnels, des variables comportementales dérivées et des labels de fraude.
# •	Nature des données : données synthétiques réalistes, générées pour imiter des comportements de clients et de fraudeurs dans un système de paiement (transactions légitimes majoritaires, fraude minoritaire).
## Type des variables et dictionnaire simplifié
Le jeu de données contient trois grandes familles de variables :
1.	Identifiants et métadonnées, 
2.	 Caractéristiques transactionnelles et contextuelles,
3.	 Variables comportementales / scores, 
4.	 Labels de fraude.
# Voici un dictionnaire des principales features (les noms exacts peuvent varier légèrement, mais la sémantique est la suivante).
# •	transaction_id
# •	Type : variable catégorielle / identifiant (string ou entier).
# •	Rôle : identifiant unique de la transaction, utile pour la traçabilité mais pas comme feature explicative directe.
# •	timestamp
# •	Type : Date/Heure (string à parser en datetime).
# •	Signification : date et heure exactes de la transaction, permettant d’extraire heure de la journée, jour de la semaine, saison, etc.
# •	sender_account_id
# •	Type : catégorielle (identifiant de compte).
# •	Signification : compte initiateur de la transaction (payer). Sert à regrouper les transactions par client et modéliser le comportement historique.
# •	receiver_account_id
# •	Type : catégorielle.
# •	Signification : compte bénéficiaire de la transaction (payee). Permet de détecter des comptes “hubs” ou des destinations suspectes récurrentes.
# •	amount
# •	Type : numérique continue (float).
# •	Signification : montant de la transaction en USD, variable centrale pour repérer des montants atypiques, seuils élevés, smurfing, etc.
# •	transaction_type
# •	Type : catégorielle (nominale).
# •	Modalités : deposit, withdrawal, transfer, payment (par exemple).
# •	Signification : nature opérationnelle de la transaction, utile pour distinguer des patterns de fraude propres aux retraits ou aux transferts.
# •	merchant_category
# •	Type : catégorielle.
# •	Signification : catégorie du commerçant ou du service (retail, utilities, services financiers, etc.), utile pour capturer des secteurs à risque plus élevé.
# •	location
# •	Type : catégorielle (pays, ville, ou code géographique).
# •	Signification : localisation déclarée de l’initiateur (ou du point de vente), permet de mesurer les écarts par rapport à la localisation habituelle du client.
# •	device_type
# •	Type : catégorielle.
# •	Modalités : mobile, web, atm, pos.
# •	Signification : canal d’initiation (smartphone, navigateur web, guichet automatique, terminal de paiement), utilisé pour capturer des schémas de fraude spécifiques à un canal.
# •	ip_address
# •	Type : catégorielle (string).
# •	Signification : IP utilisée pour la transaction, utile pour la détection de proxys, d’IP à risque ou de changements soudains d’IP.
# •	device_hash
# •	Type : catégorielle (pseudonymisée).
# •	Signification : identifiant pseudonymisé du device, permettant d’agréger l’activité par appareil tout en respectant l’anonymisation.
# •	time_since_last_transaction
# •	Type : numérique continue (e.g. secondes, minutes ou heures).
# •	Signification : délai depuis la précédente transaction du même compte, indicateur de fréquence et d’activité anormale (rafales de transactions).
# •	spending_deviation_score
# •	Type : numérique continue (score).
# •	Signification : écart du montant courant par rapport au profil historique de dépenses du client (z-score ou score normalisé). Valeur élevée = comportement inhabituel.
# •	velocity_score
# •	Type : numérique continue.
# •	Signification : score résumant la “vélocité” des transactions (nombre et volume sur une fenêtre temporelle récente), souvent corrélé aux attaques automatisées ou à la fraude en rafale.
# •	geo_anomaly_score
# •	Type : numérique continue.
# •	Signification : score d’anomalie géographique (distance ou incohérence entre la localisation actuelle et les localisations habituelles du client).
# •	is_fraud (variable cible principale)
# •	Type : binaire (0 / 1).
# •	Signification : indique si la transaction est frauduleuse (1) ou légitime (0).
# •	Rôle : target de la classification binaire standard ; le dataset est fortement déséquilibré, ce qui reflète la rareté de la fraude dans la réalité.
# •	fraud_type (variable cible secondaire / explicative)
# •	Type : catégorielle.
# •	Exemples de modalités : money laundering, account takeover, card-not-present fraud, etc.
# •	Signification : type de fraude lorsqu’is_fraud = 1 ; peut être utilisé comme cible d’un modèle multi-classe ou comme variable descriptive pour l’analyse des patterns de fraude.

