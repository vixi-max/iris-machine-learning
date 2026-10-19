# 🌸 Projet Machine Learning : Classification des fleurs Iris

Ce projet est une première introduction au **Machine Learning** avec Python. Il consiste à entraîner des modèles capables de prédire l'espèce d'une fleur Iris à partir de quatre mesures : la longueur et la largeur des sépales et des pétales.

Les trois espèces à reconnaître sont : `setosa`, `versicolor` et `virginica`.

## 📦 Installation

Installer les bibliothèques nécessaires :

```bash
pip install -r requirements.txt
```

## 🚀 Lancer le projet

Exécuter le programme :

```bash
python projet_iris.py
```

Le programme analyse les données, génère des graphiques, entraîne et évalue les modèles de classification, puis affiche leurs résultats.

## 🔍 Prédire l'espèce d'une fleur

Pour prédire l'espèce d'une fleur à partir de ses quatre mesures (en centimètres), exécuter :

```bash
python projet_iris.py --predire 6.5 3.0 5.5 2.0
```

Les valeurs correspondent, dans l'ordre, à :

1. Longueur du sépale
2. Largeur du sépale
3. Longueur du pétale
4. Largeur du pétale

Le programme affiche l'espèce prédite par le modèle.

## 📁 Contenu du projet

- `projet_iris.py` : code source complet, commenté en français.
- `requirements.txt` : liste des bibliothèques Python nécessaires.
- `graphiques/` : dossier contenant les graphiques générés lors de l'exécution, notamment les nuages de points, les corrélations et les matrices de confusion.

## 🛠️ Technologies utilisées

- **Python** : langage de programmation.
- **Pandas** : manipulation et analyse des données.
- **Matplotlib** : création de graphiques.
- **Scikit-learn** : préparation des données, entraînement et évaluation des modèles.
- **KNN et arbre de décision** : algorithmes de classification utilisés.

## 🎯 Objectifs pédagogiques

- Charger et explorer un jeu de données.
- Comprendre les variables d'entrée (*features*) et la variable cible (*target*).
- Séparer les données d'entraînement et de test.
- Entraîner et comparer plusieurs modèles de Machine Learning.
- Évaluer les performances avec l'accuracy et la matrice de confusion.
- Prédire l'espèce d'une nouvelle fleur.

**Résultat obtenu :** lors de l'exécution actuelle, le modèle KNN atteint une précision de 100 % sur les 30 fleurs de test, contre 93,33 % pour l'arbre de décision. Ces résultats concernent cette exécution et ne garantissent pas les mêmes performances sur toutes les nouvelles données.
