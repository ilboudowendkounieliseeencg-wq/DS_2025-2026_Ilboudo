# Description du Dataset Fraude Bancaire

## Aperçu général

Ce dataset correspond à un problème de **classification supervisée**, principalement de classification binaire (fraude vs non-fraude), avec une possibilité naturelle d'étendre à une classification multi-classe si l'on exploite les types de fraude disponibles. Il peut aussi servir de base à des tâches dérivées comme la détection d'anomalies non supervisée ou la modélisation temporelle de comportements de comptes bancaires.

## Problématique de modélisation

### Tâches principales et secondaires

- **Tâche principale** : prédire si une transaction donnée est frauduleuse (variable cible binaire, par exemple `is_fraud` = 0/1)
- **Tâche secondaire possible** : prédire le type de fraude (money laundering, account takeover, etc.), ce qui devient alors une classification multi-classe sur les seules transactions frauduleuses ou sur l'ensemble des transactions

### Autres usages

- Détection d'anomalies / outliers sur les scores comportementaux (velocity, geo-anomaly) et les montants, sans utiliser le label pour l'apprentissage
- Modèles séquentiels par compte (séries temporelles) pour détecter des ruptures de comportement (avant/après fraude)

### Formulation business

Dans un contexte de cabinet d'études, la formulation business typique est : *"à partir des informations disponibles au moment d'une transaction, est-il probable qu'elle soit frauduleuse, et de quel type de fraude s'agit-il ?"*

## Caractéristiques globales du dataset

| Attribut | Description |
|----------|-------------|
| **Taille** | Environ 5 000 000 lignes, chaque ligne représentant une transaction financière individuelle |
| **Période temporelle** | Transactions simulées sur une année glissante récente (2023–2024), avec une granularité fine et une répartition régulière dans le temps |
| **Structure** | Un fichier tabulaire principal (format CSV) contenant variables d'identification, attributs transactionnels, variables comportementales dérivées et labels de fraude |
| **Nature des données** | Données synthétiques réalistes, générées pour imiter des comportements de clients et de fraudeurs dans un système de paiement |
| **Déséquilibre** | Transactions légitimes majoritaires, fraude minoritaire (reflète la rareté de la fraude en réalité) |

## Familles de variables

Le jeu de données contient quatre grandes familles de variables :

1. **Identifiants et métadonnées**
2. **Caractéristiques transactionnelles et contextuelles**
3. **Variables comportementales / scores**
4. **Labels de fraude**

## Dictionnaire des principales features

### Identifiants et traçabilité

#### `transaction_id`
- **Type** : Catégorielle / identifiant (string ou entier)
- **Rôle** : Identifiant unique de la transaction, utile pour la traçabilité mais pas comme feature explicative directe

#### `timestamp`
- **Type** : Date/Heure (string à parser en datetime)
- **Signification** : Date et heure exactes de la transaction, permettant d'extraire heure de la journée, jour de la semaine, saison, etc.

### Acteurs de la transaction

#### `sender_account_id`
- **Type** : Catégorielle (identifiant de compte)
- **Signification** : Compte initiateur de la transaction (payer). Sert à regrouper les transactions par client et modéliser le comportement historique

#### `receiver_account_id`
- **Type** : Catégorielle
- **Signification** : Compte bénéficiaire de la transaction (payee). Permet de détecter des comptes "hubs" ou des destinations suspectes récurrentes

### Montant et nature de la transaction

#### `amount`
- **Type** : Numérique continue (float)
- **Signification** : Montant de la transaction en USD, variable centrale pour repérer des montants atypiques, seuils élevés, smurfing, etc.

#### `transaction_type`
- **Type** : Catégorielle (nominale)
- **Modalités** : deposit, withdrawal, transfer, payment (par exemple)
- **Signification** : Nature opérationnelle de la transaction, utile pour distinguer des patterns de fraude propres aux retraits ou aux transferts

### Contexte commercial et géographique

#### `merchant_category`
- **Type** : Catégorielle
- **Signification** : Catégorie du commerçant ou du service (retail, utilities, services financiers, etc.), utile pour capturer des secteurs à risque plus élevé

#### `location`
- **Type** : Catégorielle (pays, ville, ou code géographique)
- **Signification** : Localisation déclarée de l'initiateur (ou du point de vente), permet de mesurer les écarts par rapport à la localisation habituelle du client

### Canal et dispositif

#### `device_type`
- **Type** : Catégorielle
- **Modalités** : mobile, web, atm, pos
- **Signification** : Canal d'initiation (smartphone, navigateur web, guichet automatique, terminal de paiement), utilisé pour capturer des schémas de fraude spécifiques à un canal

#### `ip_address`
- **Type** : Catégorielle (string)
- **Signification** : IP utilisée pour la transaction, utile pour la détection de proxys, d'IP à risque ou de changements soudains d'IP

#### `device_hash`
- **Type** : Catégorielle (pseudonymisée)
- **Signification** : Identifiant pseudonymisé du device, permettant d'agréger l'activité par appareil tout en respectant l'anonymisation

### Variables comportementales et scores

#### `time_since_last_transaction`
- **Type** : Numérique continue (e.g. secondes, minutes ou heures)
- **Signification** : Délai depuis la précédente transaction du même compte, indicateur de fréquence et d'activité anormale (rafales de transactions)

#### `spending_deviation_score`
- **Type** : Numérique continue (score)
- **Signification** : Écart du montant courant par rapport au profil historique de dépenses du client (z-score ou score normalisé). Valeur élevée = comportement inhabituel

#### `velocity_score`
- **Type** : Numérique continue
- **Signification** : Score résumant la "vélocité" des transactions (nombre et volume sur une fenêtre temporelle récente), souvent corrélé aux attaques automatisées ou à la fraude en rafale

#### `geo_anomaly_score`
- **Type** : Numérique continue
- **Signification** : Score d'anomalie géographique (distance ou incohérence entre la localisation actuelle et les localisations habituelles du client)

### Variables cibles

#### `is_fraud` (variable cible principale)
- **Type** : Binaire (0 / 1)
- **Signification** : Indique si la transaction est frauduleuse (1) ou légitime (0)
- **Rôle** : Target de la classification binaire standard ; le dataset est fortement déséquilibré, ce qui reflète la rareté de la fraude dans la réalité

#### `fraud_type` (variable cible secondaire / explicative)
- **Type** : Catégorielle
- **Exemples de modalités** : money laundering, account takeover, card-not-present fraud, etc.
- **Signification** : Type de fraude lorsqu'`is_fraud` = 1 ; peut être utilisé comme target d'un modèle multi-classe ou comme variable descriptive pour l'analyse des patterns
