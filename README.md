# 🫀 Tableau de Bord de Prédiction du Risque Cardiaque

🔗 **Application en ligne** :
👉 [https://miniprojetml-ktqvaf9qhovaghcnkgzqgr.streamlit.app/](https://miniprojetml-ktqvaf9qhovaghcnkgzqgr.streamlit.app/)

---

## 📌 Présentation du projet

Ce projet consiste en le développement d’un **tableau de bord interactif basé sur le Machine Learning** permettant de **prédire le risque de maladie cardiaque (Coronary Heart Disease – CHD)** à partir de données cliniques et comportementales.

L’application couvre **l’ensemble du cycle de vie d’un projet de Data Science**, depuis l’analyse exploratoire jusqu’au déploiement d’un modèle prédictif via **Streamlit**.

---

## 🎯 Objectifs

* Réaliser une **Analyse Exploratoire des Données (EDA)**
* Construire des **pipelines de prétraitement robustes**
* Appliquer une **réduction de dimension par ACP**
* Entraîner et comparer plusieurs **modèles supervisés**
* Gérer le **déséquilibre des classes** avec SMOTE
* Sélectionner, sauvegarder et déployer le **meilleur modèle**
* Proposer une **interface interactive et intuitive**

---

## 📊 Jeu de données

Le projet utilise le **dataset CHD (Coronary Heart Disease)** contenant des données cliniques relatives à des patients.

### 🔹 Variables d’entrée

* Pression artérielle systolique (SBP)
* Consommation de tabac
* LDL cholestérol
* Adiposité
* Comportement de type A
* Obésité
* Consommation d’alcool
* Âge
* Antécédents familiaux (famhist)

### 🔹 Variable cible

* **chd**

  * `0` : Absence de maladie cardiaque
  * `1` : Présence de maladie cardiaque

---

## ⚙️ Pipeline de Machine Learning

L’implémentation repose sur des **pipelines scikit-learn**, garantissant la reproductibilité et évitant toute fuite de données.

### 🔹 Prétraitement

* Imputation des valeurs manquantes
* Standardisation des variables numériques
* Encodage One-Hot des variables catégorielles

### 🔹 Réduction de dimension

* **ACP (Analyse en Composantes Principales)**
* 90 % de variance expliquée

### 🔹 Modélisation

* Régression Logistique (avec et sans ACP)
* K-Nearest Neighbors (KNN)

### 🔹 Déséquilibre des classes

* **SMOTE (Synthetic Minority Over-sampling Technique)**

### 🔹 Optimisation

* Recherche d’hyperparamètres avec **GridSearchCV**

### 🔹 Sauvegarde

* Modèle final enregistré sous `Model.pkl`

---

## 🖥️ Application Streamlit

L’application permet à l’utilisateur de :

* Saisir les **informations cliniques d’un patient**
* Lancer une **prédiction en temps réel**
* Visualiser :

  * Le **niveau de risque cardiaque** (faible / élevé)
  * La **probabilité associée** à la prédiction

L’interface adopte un **design clair, professionnel et orienté lisibilité**.

---

## 🚀 Déploiement

L’application est déployée sur **Streamlit Cloud**.

🔗 **Accès direct** :
👉 [https://miniprojetml-ktqvaf9qhovaghcnkgzqgr.streamlit.app/](https://miniprojetml-ktqvaf9qhovaghcnkgzqgr.streamlit.app/)

---

## 🛠️ Technologies utilisées

* Python 3
* pandas, numpy
* scikit-learn
* imbalanced-learn
* joblib
* Streamlit

---

## 📁 Structure du projet

```
├── train_model.py        # Entraînement et sélection du modèle
├── app.py                # Application Streamlit
├── CHD.csv               # Jeu de données
├── Model.pkl             # Modèle entraîné
├── requirements.txt      # Dépendances
└── README.md             # Documentation
```

---

## ▶️ Exécution en local

```bash
# Installation des dépendances
pip install -r requirements.txt

# Entraînement du modèle
python train_model.py

# Lancement de l'application
streamlit run app.py
```

---

## 👥 Auteurs

Projet réalisé par :

* **Khalid Nait Ali**
* **Saad Sainane**
* **Zakaria Ftissa**
* **El Mehdi Amar**

📘 *Projet réalisé dans un cadre académique.*

---

