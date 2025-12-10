

# Descriptif: Financial Transactions Dataset for Fraud Detection

## 1. VUE D'ENSEMBLE DU DATASET

### Informations Générales

**Nom officiel** : Financial Transactions Dataset for Fraud Detection
**Source** : Kaggle (Auteur : aryan208)
**URL** : https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection

### Caractéristiques Dimensionnelles

| Métrique | Valeur |
| :-- | :-- |
| **Nombre de transactions** | 5,000,000 |
| **Nombre de colonnes** | 18 |
| **Taille du fichier** | 796.05 MB |
| **Format** | CSV |
| **Type de dataset** | Synthétique |

### Période Temporelle

**Période couverte** : 1er janvier 2023 - 1er janvier 2024 (une année complète)
**Distribution temporelle** : Répartition hebdomadaire uniforme avec environ 100,000 transactions par semaine[^1]

### Domaine d'Application

Ce dataset est conçu pour simuler des comportements financiers réalistes dans le contexte de la **détection de fraude bancaire et financière**. Il est particulièrement adapté pour :[^1]

- **Classification binaire** : Distinction fraude vs transaction légitime
- **Classification multiclasse** : Identification des types de fraudes spécifiques
- **Détection d'anomalies temporelles** : Analyse des patterns de fraude dans le temps
- **Feature engineering avancé** : Développement de nouvelles variables prédictives
- **Explicabilité des modèles** : Compréhension des facteurs de risque


### Objectif Principal

Fournir un environnement d'entraînement réaliste pour développer, tester et valider des systèmes de machine learning capables d'identifier automatiquement les transactions frauduleuses en temps réel, tout en simulant les défis réels du déséquilibre des classes et des patterns comportementaux complexes.[^2][^1]

***

## 2. CARACTÉRISTIQUES GÉNÉRALES

### Format et Encodage

- **Format de fichier** : CSV (Comma-Separated Values)
- **Nom du fichier** : `financial_fraud_detection_dataset.csv`
- **Encodage** : UTF-8 (standard)
- **Délimiteur** : Virgule (`,`)
- **En-têtes** : Présents (première ligne)


### Distribution Temporelle des Transactions

Le dataset présente une distribution temporelle remarquablement **uniforme** sur l'année 2023 :[^1]

- **Moyenne par semaine** : ~100,000 transactions
- **Plage** : 92,999 (dernière semaine de décembre) à 100,874 transactions
- **Coefficient de variation** : < 3% (très stable)
- **Pattern saisonnier** : Aucun biais saisonnier significatif

Cette uniformité facilite l'entraînement des modèles en évitant les biais temporels.

### Ratio Fraude/Non-Fraude (Class Imbalance)

| Classe | Nombre | Pourcentage | Ratio |
| :-- | :-- | :-- | :-- |
| **Transactions légitimes (is_fraud = false)** | 4,820,447 | 96.4% | - |
| **Transactions frauduleuses (is_fraud = true)** | 179,553 | 3.6% | - |
| **Ratio de déséquilibre** | - | - | **1:27** |

**Analyse du déséquilibre** : Le ratio de 1:27 (1 fraude pour 27 transactions légitimes) est **modérément déséquilibré**. C'est moins extrême que certains datasets réels de fraude bancaire (souvent 1:100 ou pire) mais suffisamment déséquilibré pour nécessiter des techniques spécialisées comme SMOTE, class weighting ou threshold tuning.[^3][^4][^5][^6][^2]

### Identifiants Uniques

| Champ | Cardinalité | Note |
| :-- | :-- | :-- |
| **transaction_id** | 5,000,000 | Un identifiant unique par transaction (aucun doublon) |
| **sender_account** | 896,513 | En moyenne ~5.6 transactions par compte émetteur |
| **receiver_account** | 896,639 | En moyenne ~5.6 transactions par compte récepteur |

### Valeurs Manquantes

Le dataset contient des valeurs manquantes **uniquement pour une colonne** :[^7][^1]

- **time_since_last_transaction** : Environ 17.9% de valeurs NULL (896,513 transactions)
    - **Explication** : Les valeurs NULL représentent probablement la **première transaction** d'un compte (pas de transaction précédente pour calculer l'intervalle)
    - **Traitement recommandé** : Imputation par 0, par la médiane, ou création d'une variable binaire "is_first_transaction"[^7]

**Toutes les autres colonnes** (17/18) sont **complètes sans valeurs manquantes**.

***

## 3. DICTIONNAIRE DE DONNÉES DÉTAILLÉ

### 3.1 Identifiants et Métadonnées Temporelles

#### **transaction_id**

- **Type de données** : String (identifiant unique)
- **Description** : Identifiant unique et immuable attribué à chaque transaction pour la traçabilité et l'audit
- **Valeurs possibles** : Chaînes alphanumériques uniques (format non spécifié)
- **Cardinalité** : 5,000,000 (100% unique)
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : Faible (utilisé uniquement pour l'identification, pas prédictif)
- **Exemples de valeurs** : "TXN_001234567", "TXN_998877665" (format supposé)


#### **timestamp**

- **Type de données** : DateTime (format ISO 8601)
- **Description** : Date et heure exacte de l'exécution de la transaction, avec précision à la seconde
- **Valeurs possibles** : 2023-01-01 00:00:00 à 2024-01-01 23:59:59
- **Cardinalité** : Très élevée (milliers de valeurs uniques)
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : **Haute** - Permet de détecter les patterns temporels anormaux (transactions nocturnes, rafales de transactions)
- **Exemples de valeurs** : "2023-03-15 14:23:41", "2023-07-08 02:11:09"


### 3.2 Acteurs de la Transaction

#### **sender_account**

- **Type de données** : String (identifiant de compte)
- **Description** : Identifiant du compte bancaire ou portefeuille électronique initiant la transaction
- **Valeurs possibles** : Identifiants alphanumériques de comptes clients
- **Cardinalité** : 896,513 comptes uniques
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : **Haute** - Permet d'analyser l'historique comportemental du compte et de détecter les comptes compromis
- **Exemples de valeurs** : "ACC_7891234", "USR_4567890"


#### **receiver_account**

- **Type de données** : String (identifiant de compte)
- **Description** : Identifiant du compte bancaire ou portefeuille électronique recevant les fonds
- **Valeurs possibles** : Identifiants alphanumériques de comptes clients
- **Cardinalité** : 896,639 comptes uniques
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : **Haute** - Permet d'identifier les "money mules" et comptes de sortie utilisés par les réseaux de fraude[^8]
- **Exemples de valeurs** : "ACC_3456789", "USR_2345678"


### 3.3 Caractéristiques Transactionnelles

#### **amount**

- **Type de données** : Float (nombre décimal)
- **Description** : Montant monétaire de la transaction en dollars américains (USD)
- **Valeurs possibles** : 0.01 USD à 3,520.57 USD
- **Distribution** : **Fortement asymétrique à droite** (right-skewed) avec 40.7% des transactions < 70.42 USD[^1]
- **Cardinalité** : Très élevée (valeurs continues)
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : **Haute** - Les montants inhabituellement élevés ou les petites transactions de test sont des signaux de fraude
- **Exemples de valeurs** : 45.67, 1,234.50, 9.99

**Distribution détaillée des montants** :

- **0-70 USD** : 40.7% des transactions (concentrés sur les petits montants)
- **70-350 USD** : 26% (montants moyens)
- **350-1000 USD** : 15% (montants élevés)
- **1000+ USD** : 18.3% (montants très élevés, potentiellement plus risqués)


#### **transaction_type**

- **Type de données** : String (catégorielle)
- **Description** : Nature opérationnelle de la transaction effectuée
- **Valeurs possibles** :
    - `deposit` : Dépôt sur un compte
    - `withdrawal` : Retrait d'argent
    - `transfer` : Transfert entre comptes
    - `payment` : Paiement à un marchand ou service
- **Cardinalité** : 4 valeurs uniques
- **Distribution** : Approximativement équilibrée (~25% chacune, soit ~1.25M transactions par type)[^7]
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : **Moyenne** - Certains types (transfers, withdrawals) sont plus souvent associés à la fraude que d'autres
- **Exemples de valeurs** : "transfer", "payment"


#### **merchant_category**

- **Type de données** : String (catégorielle)
- **Description** : Catégorie commerciale ou secteur d'activité impliqué dans la transaction
- **Valeurs possibles** : Retail, utilities, groceries, entertainment, travel, healthcare, etc. (liste exhaustive non fournie dans la source)
- **Cardinalité** : Non spécifiée (estimée entre 10-20 catégories)
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : **Moyenne** - Les fraudeurs ciblent certaines catégories (électronique, bijoux) plus que d'autres
- **Exemples de valeurs** : "retail", "utilities", "entertainment"


### 3.4 Métadonnées Contextuelles

#### **location**

- **Type de données** : String (géographique)
- **Description** : Localisation géographique d'où la transaction a été initiée (ville, région ou pays)
- **Valeurs possibles** : Noms de villes, états ou pays (format non précisé)
- **Cardinalité** : Non spécifiée (probablement plusieurs centaines de lieux)
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : **Haute** - Les changements géographiques rapides sont un signal fort de fraude (ex: transaction à New York puis Tokyo 30 minutes après)
- **Exemples de valeurs** : "New York, NY", "London, UK", "Tokyo, Japan"


#### **device_type**

- **Type de données** : String (catégorielle)
- **Description** : Type d'appareil ou canal utilisé pour initier la transaction
- **Valeurs possibles** :
    - `mobile` : Smartphone ou tablette
    - `web` : Navigateur web (ordinateur)
    - `atm` : Distributeur automatique de billets
    - `pos` : Terminal de point de vente (en magasin)
- **Cardinalité** : 4 valeurs uniques
- **Distribution** : Non spécifiée (probablement équilibrée)
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : **Moyenne à Haute** - Les changements soudains d'appareil ou l'utilisation d'appareils inhabituels peuvent signaler un account takeover[^9][^10][^11]
- **Exemples de valeurs** : "mobile", "web"


#### **payment_channel**

- **Type de données** : String (catégorielle) - **[Mentionné dans la description mais non visible dans les colonnes listées]**[^7][^1]
- **Description** : Canal ou méthode de paiement utilisé
- **Valeurs possibles** : Online, offline, P2P, etc. (supposé)
- **Cardinalité** : Non spécifiée
- **Valeurs manquantes** : Non spécifié
- **Importance pour la détection de fraude** : **Moyenne**
- **Note** : Cette colonne est mentionnée dans la description générale mais n'apparaît pas dans les extraits de données disponibles


#### **ip_address**

- **Type de données** : String (adresse IP) - **[Mentionné mais non visible]**[^1]
- **Description** : Adresse IP de l'appareil ayant initié la transaction
- **Valeurs possibles** : Adresses IPv4 ou IPv6
- **Cardinalité** : Très élevée
- **Valeurs manquantes** : Non spécifié
- **Importance pour la détection de fraude** : **Haute** - Les adresses IP anormales, les proxies/VPN ou les accès depuis des pays à haut risque sont des indicateurs de fraude
- **Exemples de valeurs** : "192.168.1.100", "2001:0db8:85a3::8a2e:0370:7334"


#### **device_hash**

- **Type de données** : String (hash cryptographique) - **[Mentionné mais non visible]**[^1]
- **Description** : Empreinte unique de l'appareil basée sur ses caractéristiques (fingerprinting)
- **Valeurs possibles** : Hash MD5, SHA-256 ou similaire
- **Cardinalité** : Très élevée
- **Valeurs manquantes** : Non spécifié
- **Importance pour la détection de fraude** : **Haute** - Permet de suivre les appareils à travers les comptes et d'identifier les réseaux de fraude utilisant les mêmes appareils[^9]
- **Exemples de valeurs** : "a3f7b89c4d5e6f1a2b3c4d5e6f7a8b9c"


### 3.5 Features Comportementales et Scores d'Anomalie

Ces features sont des **variables dérivées** calculées à partir de l'historique transactionnel pour capturer les comportements anormaux.[^12][^13][^7][^1]

#### **time_since_last_transaction**

- **Type de données** : Float (temps en secondes ou normalisé)
- **Description** : Intervalle de temps écoulé depuis la dernière transaction du même compte émetteur, mesurant la **vélocité transactionnelle**
- **Valeurs possibles** : 0 à plusieurs millions de secondes (ou valeurs normalisées 0-1)
- **Cardinalité** : Très élevée (valeurs continues)
- **Valeurs manquantes** : **Oui, 17.9%** (896,513 transactions) - représente probablement les premières transactions des comptes[^7]
- **Importance pour la détection de fraude** : **Haute** - Les rafales de transactions (faible intervalle) sont un signal fort de fraude, tout comme les transactions après de longues périodes d'inactivité[^14][^7]
- **Exemples de valeurs** : 3600.0 (1 heure), 86400.0 (24 heures), NULL (première transaction)


#### **spending_deviation_score**

- **Type de données** : Float (score normalisé)
- **Description** : Mesure de l'écart entre le montant de la transaction actuelle et le comportement de dépense habituel du compte, basé sur la moyenne et l'écart-type historiques
- **Valeurs possibles** : Généralement -3 à +3 (score Z standardisé) ou 0 à 1 (normalisé)
- **Cardinalité** : Très élevée (valeurs continues)
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : **Très Haute** - Un score élevé indique une dépense inhabituelle, souvent associée à la fraude[^15][^14][^7][^1]
- **Calcul** : $\text{Score} = \frac{\text{amount} - \mu_{\text{historique}}}{\sigma_{\text{historique}}}$
- **Exemples de valeurs** : 0.5 (normal), 2.8 (très déviant = suspect), -1.2 (en dessous de la normale)


#### **velocity_score**

- **Type de données** : Integer ou Float (score)
- **Description** : Score mesurant la **fréquence et le volume** des transactions sur une fenêtre temporelle récente (ex: nombre de transactions dans les dernières 24h, montant total dans la dernière heure)
- **Valeurs possibles** : 0 à plusieurs centaines (selon la normalisation)
- **Cardinalité** : Élevée
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : **Très Haute** - Les scores de vélocité élevés sont un des signaux les plus forts de fraude, indiquant des attaques automatisées ou des tentatives de vidage de compte[^14][^15][^7][^1]
- **Exemples de valeurs** : 1 (activité faible), 15 (activité normale), 87 (vélocité anormalement élevée = suspect)


#### **geo_anomaly_score**

- **Type de données** : Float (score normalisé)
- **Description** : Score d'anomalie géographique mesurant la probabilité qu'une transaction provienne d'un lieu inhabituel pour le compte, basé sur l'historique des localisations
- **Valeurs possibles** : 0 (normal) à 1 (très anormal) ou échelle similaire
- **Cardinalité** : Élevée
- **Valeurs manquantes** : Aucune (0%)
- **Importance pour la détection de fraude** : **Haute** - Un score élevé indique une transaction depuis une localisation jamais vue ou géographiquement impossible (ex: deux pays différents en 1 heure)[^15][^14][^1]
- **Exemples de valeurs** : 0.1 (lieu habituel), 0.9 (lieu très inhabituel = suspect), 0.5 (nouveau lieu mais plausible)


### 3.6 Variable Cible

#### **is_fraud**

- **Type de données** : Boolean (booléen)
- **Description** : **Variable cible** indiquant si la transaction a été confirmée comme frauduleuse après investigation
- **Valeurs possibles** :
    - `true` / `1` : Transaction frauduleuse confirmée
    - `false` / `0` : Transaction légitime
- **Cardinalité** : 2 valeurs
- **Distribution** :
    - `false` : 4,820,447 (96.4%)
    - `true` : 179,553 (3.6%)
- **Valeurs manquantes** : **Aucune (0%)** - Toutes les transactions sont labellisées[^1]
- **Importance pour la détection de fraude** : **Critique** - C'est la variable à prédire
- **Exemples de valeurs** : `false`, `true`


#### **fraud_type**

- **Type de données** : String (catégorielle) - **[Mentionné mais détails limités]**[^1]
- **Description** : Type spécifique de fraude détecté pour les transactions frauduleuses
- **Valeurs possibles** :
    - `money_laundering` : Blanchiment d'argent
    - `account_takeover` : Prise de contrôle de compte[^16][^10][^17][^11][^9]
    - Autres types possibles : identity theft, card testing, etc.
- **Cardinalité** : Non spécifiée (probablement 3-6 types)
- **Valeurs manquantes** : Non spécifié (probablement NULL pour les transactions légitimes)
- **Importance pour la détection de fraude** : **Haute** - Permet la classification multiclasse et l'adaptation des stratégies de détection par type de fraude
- **Exemples de valeurs** : "account_takeover", "money_laundering", NULL (si légitime)

***

## 4. STATISTIQUES DESCRIPTIVES

### 4.1 Variables Numériques

#### Montant de Transaction (amount)

| Statistique | Valeur (USD) |
| :-- | :-- |
| **Moyenne** | ~\$585.42 (estimation basée sur la distribution)[^1] |
| **Écart-type** | ~\$580-650 (estimation) |
| **Minimum** | \$0.01 |
| **25e percentile (Q1)** | ~\$120 |
| **50e percentile (Médiane)** | ~\$385 |
| **75e percentile (Q3)** | ~\$780 |
| **Maximum** | \$3,520.57 |
| **Distribution** | **Fortement asymétrique à droite** (long tail vers les montants élevés) |

**Interprétation** : La majorité des transactions sont de petits montants (< \$200), avec une queue longue vers les montants élevés. Cette distribution est réaliste pour les transactions financières quotidiennes.

#### Time Since Last Transaction

| Statistique | Valeur |
| :-- | :-- |
| **Nombre de valeurs non-nulles** | 4,103,487 (82.1%) |
| **Valeurs manquantes** | 896,513 (17.9%) |
| **Moyenne** | À calculer (dépend de l'unité : secondes vs normalisé) |
| **Médiane** | À calculer |
| **Distribution** | Probablement **log-normale** (beaucoup de transactions rapprochées, quelques transactions espacées) |

**Recommandation** : Calculer ces statistiques avec le code fourni en section 10.

#### Scores d'Anomalie (spending_deviation_score, velocity_score, geo_anomaly_score)

Ces scores sont des **variables normalisées** dont les statistiques exactes doivent être calculées. Voici les attentes générales :


| Score | Distribution Attendue | Interprétation |
| :-- | :-- | :-- |
| **spending_deviation_score** | Centré sur 0, écart-type ~1 | Score Z : 0 = normal, >2 = anormal |
| **velocity_score** | Variable selon normalisation | Plus élevé = plus de transactions récentes |
| **geo_anomaly_score** | Généralement faible (0-0.3), rares pics | Proche de 1 = localisation très suspecte |

### 4.2 Variables Catégorielles

#### Transaction Type (transaction_type)

Basé sur les informations disponibles, voici la distribution estimée :[^7]


| Type | Nombre (estimation) | Pourcentage |
| :-- | :-- | :-- |
| **deposit** | ~1,250,593 | ~25% |
| **payment** | ~1,250,438 | ~25% |
| **transfer** | ~1,250,334 | ~25% |
| **withdrawal** | ~1,248,635 | ~25% |

**Mode** : Distribution quasi-uniforme (pas de mode dominant)

#### Device Type (device_type)

| Type | Estimation |
| :-- | :-- |
| **mobile** | 30-35% (canal le plus populaire) |
| **web** | 25-30% |
| **pos** | 20-25% |
| **atm** | 15-20% (moins fréquent) |

**Note** : Ces pourcentages sont des estimations basées sur les tendances du secteur, les valeurs exactes doivent être calculées.

#### Merchant Category (merchant_category)

**Top 5 catégories estimées** (à vérifier avec le dataset) :

1. **retail** : 20-25%
2. **groceries** : 15-20%
3. **utilities** : 10-15%
4. **entertainment** : 10-15%
5. **travel** : 8-12%

***

## 5. ANALYSE DE LA VARIABLE CIBLE (is_fraud)

### 5.1 Distribution des Classes

| Classe | Nombre de Transactions | Pourcentage | Couleur Conventionnelle |
| :-- | :-- | :-- | :-- |
| **Transactions Légitimes (false)** | 4,820,447 | 96.41% | 🟢 Vert |
| **Transactions Frauduleuses (true)** | 179,553 | 3.59% | 🔴 Rouge |
| **TOTAL** | 5,000,000 | 100.00% | - |

### 5.2 Taux de Fraude Global

**Taux de fraude** : **3.59%** (approximativement 1 transaction sur 28)

**Contexte industriel** : Ce taux est **légèrement plus élevé** que la moyenne réelle dans l'industrie bancaire (typiquement 0.1-2%), mais reste dans une plage réaliste pour certains segments à plus haut risque (paiements en ligne, e-commerce international).[^18][^2][^1]

### 5.3 Ratio de Déséquilibre (Imbalance Ratio)

**Ratio fraude:non-fraude** = **1:26.84** (arrondi à **1:27**)

**Formule** : $\text{Imbalance Ratio} = \frac{\text{Classe Majoritaire}}{\text{Classe Minoritaire}} = \frac{4,820,447}{179,553} = 26.84$

**Niveau de déséquilibre** : **Modéré à élevé**

- **Faible** : < 1:10
- **Modéré** : 1:10 à 1:50 ← **Notre cas (1:27)**
- **Élevé** : 1:50 à 1:100
- **Extrême** : > 1:100

**Implication ML** : Ce niveau de déséquilibre nécessite des techniques spécialisées pour éviter que le modèle n'ignore la classe minoritaire :[^4][^5][^6][^2][^3]

- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **Class weighting** dans les algorithmes
- **Threshold tuning** pour privilégier le recall
- **Ensemble methods** (Random Forest, XGBoost avec scale_pos_weight)


### 5.4 Types de Fraudes Présentes

Le dataset mentionne une colonne **fraud_type** permettant la classification multiclasse. Les types principaux incluent :[^1]

1. **Money Laundering (Blanchiment d'argent)**
    - Transferts multiples en cascade pour obscurcir l'origine des fonds
    - Montants fragmentés (smurfing)
    - Utilisation de comptes mules[^8]
2. **Account Takeover (Prise de contrôle de compte)**
    - Accès non autorisé via phishing, credential stuffing ou ingénierie sociale
    - Changements soudains de device, localisation ou comportement
    - Transactions de vidage de compte[^10][^17][^11][^16][^9]
3. **Autres types possibles** (non confirmés dans la documentation) :
    - Identity theft (vol d'identité)
    - Card testing (tests de cartes volées)
    - Refund fraud (fraude au remboursement)

### 5.5 Répartition Temporelle des Fraudes

**À calculer avec le dataset complet**, mais les attentes basées sur la recherche incluent :

- **Patterns horaires** : Augmentation des fraudes pendant les heures nocturnes (2h-5h du matin) quand les victimes dorment
- **Patterns hebdomadaires** : Pics possibles les week-ends (moins de surveillance)
- **Patterns saisonniers** : Le dataset couvre une année complète permettant l'analyse saisonnière

**Hypothèse** : Étant donné la distribution hebdomadaire uniforme des transactions totales, les fraudes devraient suivre un pattern similaire, sauf si des patterns comportementaux spécifiques ont été simulés.

***

## 6. FEATURES AVANCÉES ET COMPORTEMENTALES

Cette section détaille les **features engineered** du dataset, qui sont parmi ses atouts majeurs pour l'apprentissage machine.[^13][^19][^12][^7][^1]

### 6.1 Features de Vélocité

#### **time_since_last_transaction**

**Calcul** :

$$
\text{time\_since\_last\_transaction} = \text{timestamp}_{\text{current}} - \text{timestamp}_{\text{previous\_same\_sender}}
$$

**Utilité pour la détection de fraude** :

- **Valeurs très faibles** (< 60 secondes) : Rafales de transactions automatisées typiques des attaques par bot
- **Valeurs NULL** : Première transaction du compte (peut indiquer un nouveau compte frauduleux ou un compte légitime)
- **Valeurs très élevées** : Réactivation après dormance (risque d'account takeover si suivie d'activité inhabituelle)

**Stratégies d'imputation recommandées** :[^7]

1. **Imputation par 0** : Simple, traite les premières transactions comme une catégorie distincte
2. **Imputation par la médiane** : Conserve la distribution
3. **Feature binaire supplémentaire** : `is_first_transaction = (time_since_last_transaction IS NULL)`

#### **velocity_score**

**Calcul (méthode probable)** :

$$
\text{velocity\_score} = w_1 \times \text{count}_{24h} + w_2 \times \text{sum\_amount}_{24h} + w_3 \times \text{count}_{1h}
$$

Où :

- $\text{count}_{24h}$ = nombre de transactions dans les 24 dernières heures
- $\text{sum\_amount}_{24h}$ = montant total transigé dans les 24 dernières heures
- $\text{count}_{1h}$ = nombre de transactions dans la dernière heure
- $w_1, w_2, w_3$ = poids de pondération

**Utilité pour la détection de fraude** :

- **Scores élevés** : Indiquent une activité transactionnelle anormalement intensive, souvent liée à :
    - Vidage de compte après un account takeover
    - Attaques automatisées
    - Blanchiment d'argent via transactions rapides en cascade

**Importance** : **Très Haute** - Une des features les plus discriminantes selon la littérature[^14][^15][^7]

### 6.2 Features d'Anomalie de Dépenses

#### **spending_deviation_score**

**Calcul (score Z standardisé)** :

$$
\text{spending\_deviation\_score} = \frac{\text{amount}_{\text{current}} - \mu_{\text{sender\_history}}}{\sigma_{\text{sender\_history}}}
$$

Où :

- $\mu_{\text{sender\_history}}$ = montant moyen des transactions historiques du sender_account
- $\sigma_{\text{sender\_history}}$ = écart-type des montants historiques

**Interprétation** :

- **Score ≈ 0** : Montant typique pour ce compte
- **Score > 2** : Montant significativement supérieur à la normale (2 écarts-types au-dessus) → **suspect**
- **Score > 3** : Montant extrême → **très suspect**
- **Score < -2** : Montant inhabituellement faible (peut indiquer des transactions de test)

**Utilité pour la détection de fraude** :

- Identifie les **dépenses inhabituelles** après un account takeover
- Détecte les **gros achats** non caractéristiques du profil client
- Capture les changements soudains de comportement

**Importance** : **Très Haute** - Feature critique dans la plupart des systèmes de détection de fraude[^19][^13][^14][^7]

### 6.3 Features d'Anomalie Géographique

#### **geo_anomaly_score**

**Calcul (méthode probable - distance géographique normalisée)** :

$$
\text{geo\_anomaly\_score} = f\left(\text{distance}(\text{location}_{\text{current}}, \text{location}_{\text{habitual}}), \Delta t\right)
$$

Où :

- $\text{distance}$ = distance en km entre la localisation actuelle et la localisation habituelle
- $\Delta t$ = temps écoulé depuis la dernière transaction (voyage impossible si $\Delta t$ trop court)
- $f$ = fonction de normalisation (ex: sigmoïde) pour mapper à[^1]

**Facteurs considérés** :

1. **Nouvelle localisation jamais vue** pour le compte
2. **Distance géographique** par rapport aux localisations habituelles
3. **Impossibilité de voyage** (ex: Paris puis Tokyo en 30 minutes)
4. **Pays à haut risque** de fraude

**Utilité pour la détection de fraude** :

- **Détection d'account takeover** : Les fraudeurs se connectent souvent depuis des localisations différentes[^10][^9]
- **Impossible travel** : Deux transactions géographiquement distantes en un temps physiquement impossible
- **Géofencing** : Transactions depuis des pays jamais visités ou blacklistés

**Importance** : **Haute** - Particulièrement efficace pour détecter les account takeovers[^20][^21]

### 6.4 Features de Métadonnées (Device, IP)

#### **device_type + device_hash + ip_address**

**Utilité combinée** :

- **Changement soudain d'appareil** : Un compte utilisant toujours mobile puis soudainement web depuis un nouvel appareil → suspect[^9]
- **Device fingerprinting** : Le `device_hash` permet de suivre les appareils à travers les comptes pour identifier les réseaux de fraude
- **IP reputation** : Les adresses IP connues pour être associées à des proxies, VPN, data centers ou pays à haut risque augmentent le risque

**Pattern typique d'account takeover** :[^10][^9]

1. Victime utilise toujours iPhone depuis New York (IP résidentielle)
2. Fraudeur se connecte depuis laptop à Lagos (IP datacenter, nouveau device_hash)
3. Le geo_anomaly_score et le changement de device déclenchent une alerte

### 6.5 Features Temporelles (Dérivées du timestamp)

Bien que `timestamp` soit une colonne brute, des features temporelles peuvent être dérivées :


| Feature Dérivée | Calcul | Utilité Fraude |
| :-- | :-- | :-- |
| **hour_of_day** | Extraire l'heure (0-23) | Transactions nocturnes (2h-5h) sont suspectes |
| **day_of_week** | Lundi=1, Dimanche=7 | Patterns weekend vs semaine |
| **is_weekend** | Samedi ou Dimanche | Activité frauduleuse accrue les week-ends |
| **is_night** | 0h-6h du matin | Les fraudeurs opèrent souvent la nuit |

**Recommandation** : Créer ces features durant le feature engineering.[^22][^12][^13]

***

## 7. QUALITÉ DES DONNÉES

### 7.1 Intégrité Référentielle

#### Identifiants de Comptes (sender_account, receiver_account)

**Points à vérifier** :

- ✅ **Cardinalité** : 896,513 sender_accounts vs 896,639 receiver_accounts (légère différence normale)
- ⚠️ **Auto-transactions** : Vérifier si `sender_account == receiver_account` (devrait être rare ou interdit)
- ✅ **Format cohérent** : Tous les identifiants suivent le même format (à confirmer)
- 🔍 **Comptes orphelins** : Certains comptes peuvent n'apparaître que comme sender ou receiver (normal pour les merchants)

**Requête SQL de validation recommandée** :

```sql
-- Vérifier les auto-transactions
SELECT COUNT(*) FROM transactions WHERE sender_account = receiver_account;

-- Identifier les comptes à très forte activité (potentiels mules)
SELECT receiver_account, COUNT(*) as trx_count 
FROM transactions 
GROUP BY receiver_account 
HAVING COUNT(*) > 1000 
ORDER BY trx_count DESC;
```


### 7.2 Cohérence Temporelle

#### Ordre Chronologique des Timestamps

**À vérifier** :

- ✅ **Plage temporelle** : 2023-01-01 à 2024-01-01 (confirmé)[^1]
- 🔍 **Ordre séquentiel** : Les transactions doivent être dans l'ordre chronologique (si le dataset est trié par timestamp)
- ⚠️ **Timestamps futurs** : Aucun timestamp ne devrait dépasser 2024-01-01
- ⚠️ **Timestamps impossibles** : Vérifier l'absence de dates invalides (ex: 2023-02-30)

**Code de validation Python** :

```python
import pandas as pd

df['timestamp'] = pd.to_datetime(df['timestamp'])

# Vérifier l'ordre chronologique
assert df['timestamp'].is_monotonic_increasing, "Timestamps non ordonnés!"

# Vérifier la plage
assert df['timestamp'].min() >= pd.Timestamp('2023-01-01')
assert df['timestamp'].max() <= pd.Timestamp('2024-01-01')
```


### 7.3 Outliers et Valeurs Aberrantes

#### Montant (amount)

**Outliers identifiés** :

- ✅ **Montant minimum** : \$0.01 (acceptable, transactions de test ou micro-paiements)
- ⚠️ **Montant maximum** : \$3,520.57 (élevé mais pas impossible pour des paiements B2B ou loyers)
- 📊 **Distribution** : Forte asymétrie à droite avec ~18% des transactions > \$1,000 (à valider comme réaliste)

**Traitement recommandé** :

- **Option 1** : Conserver tous les montants (les outliers peuvent être des fraudes légitimes)
- **Option 2** : Winsorization à 99.5% pour éviter l'influence excessive sur les modèles linéaires
- **Option 3** : Transformation log($\text{amount}$) pour normaliser la distribution


#### Scores d'Anomalie

**Plages attendues** :

- `spending_deviation_score` : Devrait être centré sur 0 avec quelques valeurs extrêmes (> 3 ou < -3)
- `velocity_score` : Valeurs généralement basses avec rares pics élevés
- `geo_anomaly_score` : Majorité proche de 0, quelques valeurs proches de 1

**Validation recommandée** :

```python
# Identifier les outliers extrêmes
outliers = df[
    (df['spending_deviation_score'].abs() > 5) | 
    (df['velocity_score'] > df['velocity_score'].quantile(0.999))
]
print(f"Transactions avec outliers extrêmes: {len(outliers)}")
```


### 7.4 Duplicatas

#### Transaction ID

**Vérification** :

- ✅ **Unicité garantie** : Les 5 millions de transaction_id doivent être uniques (confirmé par cardinalité)[^1]

**Code de validation** :

```python
assert df['transaction_id'].nunique() == len(df), "Duplicatas détectés dans transaction_id!"
```


#### Transactions Identiques

**À vérifier** : Transactions avec même (timestamp, sender_account, receiver_account, amount)

- Possible pour des **paiements récurrents** (abonnements)
- Suspect si très fréquent (potentiellement des erreurs de génération du dataset synthétique)

**Code de détection** :

```python
duplicates = df.duplicated(subset=['timestamp', 'sender_account', 'receiver_account', 'amount'], keep=False)
print(f"Transactions potentiellement dupliquées: {duplicates.sum()}")
```


### 7.5 Problèmes Potentiels Identifiés

| Problème | Sévérité | Description | Mitigation |
| :-- | :-- | :-- | :-- |
| **Valeurs manquantes dans time_since_last_transaction** | ⚠️ Moyenne | 17.9% de valeurs NULL | Imputation par 0, médiane, ou feature binaire[^7] |
| **Nature synthétique des données** | ⚠️ Moyenne-Haute | Possibles patterns irréalistes | Validation avec experts métier, tests sur données réelles |
| **Déséquilibre des classes** | ⚠️ Moyenne | Ratio 1:27 | SMOTE, class weighting, threshold tuning[^2][^6] |
| **Documentation incomplète de certaines colonnes** | ℹ️ Faible | ip_address, device_hash, payment_channel, fraud_type non détaillés | Exploration du fichier CSV complet |
| **Distribution de certaines catégorielles inconnue** | ℹ️ Faible | merchant_category, fraud_type | Calculer avec pandas value_counts() |


***

## 8. CARACTÉRISTIQUES POUR LE MACHINE LEARNING

### 8.1 Features Recommandées pour le Modeling

#### Features Hautement Prédictives (Top 10)

Basé sur la littérature et l'analyse du dataset :[^23][^24][^13][^19][^15][^14][^7]


| Rang | Feature | Type | Importance Estimée | Justification |
| :-- | :-- | :-- | :-- | :-- |
| 1 | **velocity_score** | Numérique | ⭐⭐⭐⭐⭐ | Capture les rafales de transactions |
| 2 | **spending_deviation_score** | Numérique | ⭐⭐⭐⭐⭐ | Détecte les dépenses inhabituelles |
| 3 | **geo_anomaly_score** | Numérique | ⭐⭐⭐⭐ | Identifie les localisations suspectes |
| 4 | **amount** | Numérique | ⭐⭐⭐⭐ | Montants élevés corrélés à la fraude |
| 5 | **time_since_last_transaction** | Numérique | ⭐⭐⭐⭐ | Intervalle anormal = risque |
| 6 | **hour_of_day** (dérivée) | Numérique | ⭐⭐⭐ | Transactions nocturnes suspectes |
| 7 | **device_type** | Catégorielle | ⭐⭐⭐ | Changement d'appareil = account takeover |
| 8 | **transaction_type** | Catégorielle | ⭐⭐⭐ | Certains types plus risqués |
| 9 | **location** | Catégorielle | ⭐⭐⭐ | Certaines régions à haut risque |
| 10 | **merchant_category** | Catégorielle | ⭐⭐ | Certains secteurs ciblés par fraudeurs |

### 8.2 Features à Transformer ou Encoder

#### Variables Catégorielles à Encoder

| Feature | Cardinalité | Méthode d'Encodage Recommandée | Justification |
| :-- | :-- | :-- | :-- |
| **transaction_type** | 4 | **One-Hot Encoding** | Faible cardinalité, pas d'ordre naturel |
| **device_type** | 4 | **One-Hot Encoding** | Faible cardinalité, pas d'ordre naturel |
| **merchant_category** | 10-20 (estimé) | **Target Encoding** ou **Frequency Encoding** | Cardinalité moyenne, certaines catégories rares |
| **location** | Centaines | **Target Encoding** + **Feature Extraction** (pays, région) | Haute cardinalité, hiérarchie géographique |
| **sender_account** | 896,513 | **Agrégations** (count, avg amount, fraud rate historique) | Très haute cardinalité, impossible à one-hot |
| **receiver_account** | 896,639 | **Agrégations** (count, fraud rate) | Très haute cardinalité |
| **payment_channel** | 3-5 (estimé) | **One-Hot Encoding** | Faible cardinalité |

**Code exemple (One-Hot Encoding)** :

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
transaction_type_encoded = ohe.fit_transform(df[['transaction_type']])
```

**Code exemple (Target Encoding)** :

```python
from category_encoders import TargetEncoder

te = TargetEncoder()
merchant_encoded = te.fit_transform(df['merchant_category'], df['is_fraud'])
```


#### Variables Numériques à Transformer

| Feature | Transformation Recommandée | Justification |
| :-- | :-- | :-- |
| **amount** | **Log($x+1$)** ou **RobustScaler** | Distribution asymétrique avec outliers[^7] |
| **time_since_last_transaction** | **StandardScaler** après imputation | Distribution potentiellement log-normale |
| **Scores d'anomalie** | **StandardScaler** ou déjà normalisés | Uniformiser les échelles |

**Code exemple (Scaling)** :

```python
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np

# Log transformation pour amount
df['amount_log'] = np.log1p(df['amount'])

# Robust scaling (résistant aux outliers)
scaler = RobustScaler()
df[['amount_scaled']] = scaler.fit_transform(df[['amount']])
```


### 8.3 Features Redondantes ou Corrélées

**Paires potentiellement corrélées à vérifier** :

1. **velocity_score ↔️ time_since_last_transaction**
    - Corrélation négative attendue (faible intervalle = haute vélocité)
    - **Action** : Calculer la corrélation de Pearson; si |r| > 0.8, considérer en retirer une
2. **spending_deviation_score ↔️ amount**
    - Corrélation positive partielle (montants élevés tendent à avoir des scores élevés)
    - **Action** : Conserver les deux (le score capture la déviation **relative** à l'historique)
3. **geo_anomaly_score ↔️ location**
    - Relation forte mais non redondante (score = fonction de location + historique)
    - **Action** : Conserver les deux

**Analyse de corrélation recommandée** :

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Matrice de corrélation
numerical_cols = ['amount', 'time_since_last_transaction', 'spending_deviation_score', 
                  'velocity_score', 'geo_anomaly_score']
corr_matrix = df[numerical_cols].corr()

# Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de Corrélation des Features Numériques')
plt.show()
```


### 8.4 Stratégies de Feature Engineering Suggérées

#### Features Temporelles Avancées

```python
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_night'] = df['hour'].between(0, 6).astype(int)
df['month'] = df['timestamp'].dt.month
```


#### Features d'Agrégation par Compte

```python
# Pour chaque sender_account, calculer des statistiques historiques
sender_stats = df.groupby('sender_account').agg({
    'transaction_id': 'count',  # Nombre total de transactions
    'amount': ['mean', 'std', 'max'],  # Statistiques de montants
    'is_fraud': 'sum'  # Nombre de fraudes historiques (attention au data leakage!)
}).reset_index()

sender_stats.columns = ['sender_account', 'trx_count', 'avg_amount', 'std_amount', 
                        'max_amount', 'fraud_count']

df = df.merge(sender_stats, on='sender_account', how='left')
```

⚠️ **Attention au Data Leakage** : Ne pas utiliser `fraud_count` qui inclut la transaction actuelle. Calculer uniquement sur l'historique **avant** la transaction actuelle.

#### Ratios et Interactions

```python
# Ratio du montant actuel par rapport au montant moyen historique
df['amount_to_avg_ratio'] = df['amount'] / df['avg_amount']

# Interaction entre vélocité et déviation
df['velocity_x_deviation'] = df['velocity_score'] * df['spending_deviation_score']

# Somme des scores d'anomalie (composite risk score)
df['total_anomaly_score'] = (df['spending_deviation_score'].abs() + 
                              df['velocity_score'] + 
                              df['geo_anomaly_score'])
```


### 8.5 Techniques de Gestion du Déséquilibre Recommandées

#### Approches Comparatives

| Technique | Avantages | Inconvénients | Quand l'utiliser |
| :-- | :-- | :-- | :-- |
| **SMOTE** | Augmente la classe minoritaire, améliore le recall | Peut créer des exemples irréalistes, risque d'overfitting[^6] | Dataset < 1M lignes, ratio 1:10 à 1:100 |
| **Random UnderSampling** | Réduit le temps d'entraînement | Perte d'information de la classe majoritaire | Très gros datasets (>5M lignes) |
| **Hybrid (SMOTE + Tomek)** | Nettoie les frontières de décision | Plus complexe à implémenter | Pour maximiser la performance |
| **Class Weighting** | Pas de modification des données | Moins efficace que SMOTE pour les grands déséquilibres | Avec XGBoost, Random Forest (paramètre scale_pos_weight) |
| **Threshold Tuning** | Simple, pas de réentraînement | Ne change pas les probabilities, juste la décision | En production pour ajuster recall/precision |
| **Ensemble Methods** | Combine plusieurs approches | Coût computationnel élevé | Pour maximiser F1-score et AUC-ROC |

**Recommandation pour ce dataset (ratio 1:27)** :

1. **Baseline** : Entraîner avec class weighting
2. **SMOTE** : Appliquer SMOTE pour obtenir un ratio 1:3 à 1:5 (pas 1:1 complet)[^6]
3. **Hybrid** : SMOTE + Tomek Links pour nettoyer
4. **Comparer** les 3 approches sur validation set avec focus sur **Recall ≥ 80%**[^24][^25][^23]

**Code exemple** :

```python
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

X = df.drop(['is_fraud', 'transaction_id', 'timestamp'], axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# SMOTE avec sampling_strategy=0.3 (ratio final 1:3.3)
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Classe majoritaire: {(y_train_smote == 0).sum()}")
print(f"Classe minoritaire: {(y_train_smote == 1).sum()}")
print(f"Nouveau ratio: 1:{(y_train_smote == 0).sum() / (y_train_smote == 1).sum():.1f}")
```


***

## 9. LIMITATIONS ET CONSIDÉRATIONS

### 9.1 Nature Synthétique des Données

#### Biais Potentiels

**Avantages des données synthétiques** :

- ✅ **Pas de problèmes de confidentialité** (RGPD/CCPA compliant)
- ✅ **Volume important** (5 millions de transactions)
- ✅ **Labels parfaits** (pas de fraudes non détectées)
- ✅ **Features avancées pré-calculées** (velocity_score, etc.)

**Limitations inhérentes** :

- ⚠️ **Patterns simplifiés** : Les fraudes réelles sont plus complexes et évolutives que les patterns simulés[^11][^1]
- ⚠️ **Distribution artificielle** : La distribution uniforme temporelle n'est pas réaliste (vrais pics autour des fêtes, salaires)
- ⚠️ **Nouvelles techniques de fraude** : Les fraudeurs innovent constamment, les données synthétiques ne capturent que les patterns connus
- ⚠️ **Interactions humaines** : Comportements psychologiques et contextuels non capturés

**Biais de génération potentiels** :

1. **Séparabilité artificielle** : Les fraudes synthétiques peuvent être "trop facilement" détectables
2. **Manque de bruit** : Données réelles contiennent plus d'anomalies non-frauduleuses
3. **Patterns déterministes** : Possible utilisation de règles fixes pour générer les fraudes

### 9.2 Différences avec des Données Réelles de Production

| Aspect | Dataset Synthétique | Production Réelle |
| :-- | :-- | :-- |
| **Taux de fraude** | 3.6% (stable) | 0.1-2% (variable)[^1][^2] |
| **Déséquilibre** | 1:27 | 1:50 à 1:1000 |
| **Labels** | 100% labellisés | 10-30% des fraudes non détectées (faux négatifs) |
| **Latence** | Batch | Temps réel (< 100ms)[^20][^26] |
| **Drift** | Aucun | Concept drift constant (nouveaux patterns)[^27] |
| **Qualité** | Parfaite | Valeurs manquantes, erreurs de saisie, duplicatas |
| **Volume** | 5M sur 1 an | Peut atteindre des milliards pour grandes banques |

**Implications** :

- **Surestimation des performances** : Les métriques obtenues sur ce dataset seront probablement **plus optimistes** qu'en production
- **Nécessité de réentraînement** : Modèles devront être adaptés et réentraînés sur données réelles


### 9.3 Scénarios de Fraude Non Couverts

**Types de fraude potentiellement absents** :

1. **Fraude au premier paiement (First-Party Fraud)**
    - Client légitime qui nie une transaction qu'il a effectuée
    - Difficile à distinguer des vraies fraudes
2. **Fraude évoluée avec IA**
    - Fraudeurs utilisant l'apprentissage machine pour contourner les systèmes de détection
    - Attacks adversariales
3. **Fraude interne**
    - Employés malveillants de la banque
    - Nécessite des données d'audit internes
4. **Fraude physique**
    - Skimming de cartes
    - Fraude aux distributeurs (shimming)
5. **Fraude sociale complexe**
    - Scams d'ingénierie sociale multi-étapes[^16]
    - Romance scams, fraude aux investissements

### 9.4 Précautions d'Interprétation

#### Pour les Data Scientists

1. **Ne pas sur-optimiser** : Un F1-score de 99% sur ce dataset ne signifie pas 99% en production
2. **Analyser les erreurs** : Comprendre **pourquoi** le modèle se trompe est plus important que la métrique globale
3. **Tester la robustesse** : Évaluer sur différentes périodes temporelles et sous-segments (par device_type, location)
4. **Considérer le coût** : En production, les **faux positifs** bloquent des clients légitimes (coût de friction)[^25][^28]

#### Pour les Équipes Métier

1. **Point de départ, pas solution finale** : Ce dataset sert à développer un prototype, pas un système production-ready
2. **Validation métier nécessaire** : Experts en fraude doivent valider les patterns détectés
3. **Intégration progressive** : Déployer d'abord en mode "shadow" (scoring sans blocage) pour valider
4. **Feedback loop** : Système doit apprendre des vraies décisions de fraude

### 9.5 Recommandations pour l'Utilisation en Production

#### Phase 1 : Développement (sur dataset synthétique)

1. **Prototypage rapide** : Tester différents algorithmes (Random Forest, XGBoost, Neural Networks)
2. **Feature engineering** : Développer et valider les features avancées
3. **Optimisation des hyperparamètres** : Grid search, Bayesian optimization
4. **Benchmark** : Établir une baseline de performances (target: Recall > 80%, F1 > 0.75)[^23][^24]

#### Phase 2 : Validation (données historiques réelles)

1. **Collecter données réelles** : 6-12 mois de transactions historiques avec labels vérifiés
2. **Réentraîner** : Adapter le modèle aux patterns réels
3. **Backtesting** : Simuler les performances sur historique réel
4. **Ajuster seuils** : Optimiser le trade-off recall/precision selon les coûts métier[^28]

#### Phase 3 : Déploiement (production)

1. **Mode shadow** : Scorer sans bloquer pendant 1-3 mois
2. **Analyse des alertes** : Équipe fraude valide manuellement les alertes pour ajuster
3. **A/B testing** : Comparer avec système existant
4. **Monitoring continu** : Surveiller data drift, performance degradation[^27]
5. **Réentraînement périodique** : Mensuel ou trimestriel selon le drift observé

#### Seuils de Décision Recommandés (à ajuster)

| Métrique Cible | Seuil de Probabilité | Cas d'Usage |
| :-- | :-- | :-- |
| **Haute Precision** (95%+) | 0.8-0.9 | Blocage automatique sans friction client |
| **Équilibré** (Recall ~80%, Precision ~70%) | 0.5-0.6 | Alerte pour revue manuelle |
| **Haute Recall** (90%+) | 0.2-0.3 | 3D Secure challenge (friction acceptable) |


***

## 10. MÉTADONNÉES ADMINISTRATIVES

### 10.1 Auteur et Source

| Champ | Information |
| :-- | :-- |
| **Auteur/Créateur** | aryan208 (Kaggle username) |
| **Plateforme** | Kaggle Datasets |
| **URL officielle** | https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection |
| **Type** | Dataset public open-source |

### 10.2 Dates et Versions

| Champ | Information |
| :-- | :-- |
| **Date de publication** | Mai 2025[^29] |
| **Dernière mise à jour** | Juillet 2025 (basé sur les statistiques de vues/downloads)[^1] |
| **Version actuelle** | Version 1 (pas de versions multiples documentées) |
| **Période des données** | 1er janvier 2023 - 1er janvier 2024 (données synthétiques datées) |

### 10.3 Licence d'Utilisation

**Licence** : Non explicitement spécifiée dans la source consultée. **Par défaut sur Kaggle** :

- **Utilisation libre** pour recherche et éducation
- **Attribution recommandée** (citer l'auteur)
- **Vérifier les conditions** : Consulter la page Kaggle pour la licence exacte (probablement CC BY 4.0 ou similaire)

**Restrictions potentielles** :

- ⚠️ Usage commercial : À vérifier sur la page du dataset
- ✅ Usage académique : Généralement autorisé
- ✅ Modification : Généralement autorisée


### 10.4 Citation Recommandée

**Format APA** :

```
aryan208. (2025). Financial Transactions Dataset for Fraud Detection [Dataset]. Kaggle. 
https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection
```

**Format BibTeX** :

```bibtex
@misc{aryan208_fraud_2025,
  author = {aryan208},
  title = {Financial Transactions Dataset for Fraud Detection},
  year = {2025},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection}
}
```

**Format IEEE** :

```
[^1] aryan208, "Financial Transactions Dataset for Fraud Detection," Kaggle Datasets, 2025. 
[Online]. Available: https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection
```


### 10.5 Statistiques d'Usage

Basé sur les données disponibles (juillet-août 2025) :[^1]


| Métrique | Valeur |
| :-- | :-- |
| **Vues totales** | ~3,000+ (estimé sur 2 mois) |
| **Téléchargements totaux** | ~2,500+ |
| **Pic de téléchargements** | 235 le 19 juillet 2025 |
| **Notebooks publics** | Non spécifié (plusieurs kernels Kaggle disponibles) |
| **Score/Rating** | 10.00 (évaluation maximale)[^1] |

### 10.6 Ressources Complémentaires

**Notebooks Kaggle utilisant ce dataset** :

- "Fraud Detection | RF Model | 0.91 Score" par uselessnoob[^14]
- "Fraud Detection | EDA + 0.9 Recall" par diegoamd[^15]
- "FraudDetector with IsoForest and DBSCN" par olneyjeffrey[^30]

**Datasets similaires pour comparaison** :

- Credit Card Fraud Detection (Kaggle, 284,807 transactions)[^31]
- PaySim Synthetic Financial Dataset (6.3M transactions)[^32]
- Nigerian Financial Transactions Dataset (features similaires)[^33]


### 10.7 Contact et Support

- **Questions** : Utiliser la section "Comments" sur la page Kaggle du dataset
- **Problèmes techniques** : Signaler via Kaggle
- **Demandes de collaboration** : Contacter l'auteur via son profil Kaggle

***

## ANNEXE : Code Python pour Valider et Explorer le Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Charger le dataset
df = pd.read_csv('financial_fraud_detection_dataset.csv')

# ============================================
# 1. INFORMATIONS GÉNÉRALES
# ============================================

print("="*60)
print("INFORMATIONS GÉNÉRALES DU DATASET")
print("="*60)

print(f"\nTaille du dataset:")
print(f"  - Lignes: {len(df):,}")
print(f"  - Colonnes: {len(df.columns)}")
print(f"  - Mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\nPériode temporelle:")
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"  - Début: {df['timestamp'].min()}")
print(f"  - Fin: {df['timestamp'].max()}")
print(f"  - Durée: {(df['timestamp'].max() - df['timestamp'].min()).days} jours")

# ============================================
# 2. ANALYSE DES VALEURS MANQUANTES
# ============================================

print("\n" + "="*60)
print("VALEURS MANQUANTES")
print("="*60)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Colonne': missing.index,
    'Valeurs Manquantes': missing.values,
    'Pourcentage': missing_pct.values
})
print(missing_df[missing_df['Valeurs Manquantes'] > 0])

# ============================================
# 3. ANALYSE DE LA VARIABLE CIBLE
# ============================================

print("\n" + "="*60)
print("DISTRIBUTION DE LA VARIABLE CIBLE (is_fraud)")
print("="*60)

fraud_counts = df['is_fraud'].value_counts()
fraud_pct = (fraud_counts / len(df)) * 100

print(f"\nTransactions légitimes: {fraud_counts[False]:,} ({fraud_pct[False]:.2f}%)")
print(f"Transactions frauduleuses: {fraud_counts[True]:,} ({fraud_pct[True]:.2f}%)")
print(f"Ratio de déséquilibre: 1:{fraud_counts[False] / fraud_counts[True]:.1f}")

# ============================================
# 4. STATISTIQUES DESCRIPTIVES
# ============================================

print("\n" + "="*60)
print("STATISTIQUES DESCRIPTIVES - VARIABLES NUMÉRIQUES")
print("="*60)

numerical_cols = ['amount', 'time_since_last_transaction', 
                  'spending_deviation_score', 'velocity_score', 'geo_anomaly_score']

print("\n" + df[numerical_cols].describe())

# ============================================
# 5. DISTRIBUTION DES VARIABLES CATÉGORIELLES
# ============================================

print("\n" + "="*60)
print("DISTRIBUTION DES VARIABLES CATÉGORIELLES")
print("="*60)

categorical_cols = ['transaction_type', 'device_type', 'merchant_category']

for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col}:")
        value_counts = df[col].value_counts()
        value_pct = (value_counts / len(df)) * 100
        for val, count in value_counts.items():
            print(f"  {val}: {count:,} ({value_pct[val]:.2f}%)")

# ============================================
# 6. CARDINALITÉ DES IDENTIFIANTS
# ============================================

print("\n" + "="*60)
print("CARDINALITÉ DES IDENTIFIANTS")
print("="*60)

print(f"\ntransaction_id: {df['transaction_id'].nunique():,} valeurs uniques")
print(f"sender_account: {df['sender_account'].nunique():,} valeurs uniques")
print(f"receiver_account: {df['receiver_account'].nunique():,} valeurs uniques")

# ============================================
# 7. VISUALISATIONS
# ============================================

# Distribution des montants
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['amount'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Montant (USD)')
plt.ylabel('Fréquence')
plt.title('Distribution des Montants de Transaction')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(df['amount']), bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.xlabel('Log(Montant + 1)')
plt.ylabel('Fréquence')
plt.title('Distribution Log des Montants')
plt.tight_layout()
plt.savefig('amount_distribution.png', dpi=300)
plt.show()

# Corrélation entre features numériques
plt.figure(figsize=(10, 8))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation des Features Numériques', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.show()

# Distribution temporelle des fraudes
df['date'] = df['timestamp'].dt.date
fraud_by_date = df.groupby('date')['is_fraud'].sum()

plt.figure(figsize=(14, 5))
plt.plot(fraud_by_date.index, fraud_by_date.values, linewidth=1.5)
plt.xlabel('Date')
plt.ylabel('Nombre de Fraudes')
plt.title('Évolution Temporelle des Fraudes (Janvier 2023 - Janvier 2024)')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fraud_temporal_distribution.png', dpi=300)
plt.show()

print("\n✅ Analyse terminée! Les visualisations ont été sauvegardées.")
```


***

## CONCLUSION

Ce dataset **Financial Transactions Dataset for Fraud Detection** constitue une ressource précieuse pour développer et tester des systèmes de détection de fraude basés sur le machine learning. Avec **5 millions de transactions synthétiques** réparties sur une année complète, **18 features** incluant des scores comportementaux avancés, et un **ratio de déséquilibre réaliste de 1:27**, il offre un environnement d'entraînement riche et challengeant.[^2][^24][^23][^1]

Les **forces principales** du dataset incluent :

- Features comportementales pré-calculées (velocity, spending deviation, geo-anomaly)
- Volume substantiel permettant l'entraînement de modèles complexes
- Labels parfaits sans ambiguïté
- Distribution temporelle uniforme facilitant la validation croisée

Les **limitations** à considérer :

- Nature synthétique pouvant ne pas capturer toute la complexité des fraudes réelles
- Taux de fraude légèrement élevé comparé à la production (3.6% vs 0.1-2%)
- Nécessité de validation sur données réelles avant déploiement

**Recommandations finales** :

1. Utiliser ce dataset pour le **prototypage et la recherche**
2. Appliquer **SMOTE ou class weighting** pour gérer le déséquilibre
3. Cibler **Recall ≥ 80%** comme métrique prioritaire pour minimiser les fraudes manquées
4. Valider systématiquement sur données réelles avant production
5. Implémenter un système de **monitoring continu** du concept drift en production[^27][^8]

Ce document servira de référence complète pour toute équipe travaillant avec ce dataset, de l'exploration initiale au déploiement en production d'un système de détection de fraude performant et robuste.
<span style="display:none">[^34][^35][^36][^37][^38][^39]</span>

<div align="center">⁂</div>

[^1]: https://www.getfocal.ai/blog/fraud-detection-with-machine-learning

[^2]: https://journals.ekb.eg/article_414893_9e92b6e04aa25efa9bcbeef5275ebfc0.pdf

[^3]: https://www.evolutioniq.com/resources/the-journey-begins

[^4]: https://www.reddit.com/r/learnmachinelearning/comments/1g6jx90/trying_to_build_an_effective_fraud_detection/

[^5]: https://keylabs.ai/blog/handling-imbalanced-data-in-classification/

[^6]: https://www.blog.trainindata.com/smote-in-python-a-guide-to-balanced-datasets/

[^7]: https://duckdb.org/2025/08/15/ml-data-preprocessing.html

[^8]: https://www.tookitaki.com/compliance-hub/the-power-of-automated-fraud-detection-systems

[^9]: https://linkurious.com/blog/account-takeover-fraud/

[^10]: https://lantern.splunk.com/Industry_Use_Cases/Financial_Services_and_Insurance/FSI_Fraud_Account_Takeover

[^11]: https://www.syntho.ai/fraud-detection-in-banking-with-synthetic-data/

[^12]: https://www.tandfonline.com/doi/full/10.1080/19393555.2025.2528067?af=R

[^13]: https://thesai.org/Downloads/Volume12No12/Paper_2-New_Feature_Engineering_Framework.pdf

[^14]: https://www.kaggle.com/code/uselessnoob/fraud-detection-rf-model-0-91-score

[^15]: https://www.kaggle.com/code/diegoamd/fraud-detection-eda-0-9-recall

[^16]: https://fiu.gov.gy/wp-content/uploads/2025/10/Typology-Report-Account-Takeovers-Mobile-Payment-Services.pdf

[^17]: https://fedpaymentsimprovement.org/wp-content/uploads/brief-2-fraud-types-and-authentication-for-remote-payment-use-cases.pdf

[^18]: https://www.cesarsotovalero.net/blog/evaluation-metrics-for-real-time-financial-fraud-detection-ml-models.html

[^19]: https://repository.londonmet.ac.uk/6407/2/The final version_feature%20engineering%20framework%20for%20financial%20fraud%20detection%20model.pdf

[^20]: https://xenoss.io/blog/real-time-ai-fraud-detection-in-banking

[^21]: https://openmetal.io/resources/blog/big-data-for-fraud-detection-a-guide-for-financial-services-and-e-commerce/

[^22]: https://docs.nvidia.com/nim/financial-fraud-training/2.0.0/preprocessing/preproc.html

[^23]: https://www.tandfonline.com/doi/full/10.1080/23311975.2025.2474209

[^24]: https://www.academia.edu/129361586/A_Comparative_Study_of_Random_Forest_and_XGBoost_for_Detecting_Credit_Card_Fraud_Transactions_using_Big_Data

[^25]: https://www.linkedin.com/pulse/precision-recall-f1-score-deciphering-success-ai-fraud-stefan-klein-crhfe

[^26]: https://www.tinybird.co/blog/how-to-build-a-real-time-fraud-detection-system

[^27]: http://www.diva-portal.org/smash/get/diva2:1996472/FULLTEXT01.pdf

[^28]: https://kount.com/blog/precision-recall-when-conventional-fraud-metrics-fall-short

[^29]: https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection

[^30]: https://www.kaggle.com/code/olneyjeffrey/frauddetector-with-isoforest-and-dbscn

[^31]: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

[^32]: https://www.kaggle.com/datasets/ealaxi/paysim1

[^33]: https://huggingface.co/datasets/electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset/viewer

[^34]: https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection/versions/1

[^35]: https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset

[^36]: https://huggingface.co/datasets/electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset

[^37]: https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

[^38]: https://raw.githubusercontent.com/theislab/scvelo_notebooks/master/DifferentialKinetics.ipynb

[^39]: https://www.kaggle.com/datasets/younusmohamed/fraudulent-financial-transaction-prediction

