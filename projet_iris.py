
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

GRAPHIQUES = "graphiques"
GRAINE = 42


def charger_donnees():
    """Charge le jeu de données Iris."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["espece"] = df["target"].map(dict(enumerate(iris.target_names)))
    return iris, df


def explorer(df):
    """Affiche un aperçu et des statistiques."""
    print("=== Aperçu des 5 premières lignes ===")
    print(df.head(), "\n")

    print("=== Statistiques descriptives ===")
    print(df.describe().round(2), "\n")

    print("=== Répartition des classes ===")
    print(df["espece"].value_counts(), "\n")

    print("=== Valeurs manquantes ===")
    print(df.isna().sum().sum(), "valeur(s) manquante(s)\n")


def visualiser(df, noms_features):
    """Génère les graphiques d'exploration."""
    os.makedirs(GRAPHIQUES, exist_ok=True)

    plt.figure(figsize=(6, 4.5))

    for espece, groupe in df.groupby("espece"):
        plt.scatter(
            groupe["petal length (cm)"],
            groupe["petal width (cm)"],
            label=espece,
        )

    plt.xlabel("Longueur du pétale (cm)")
    plt.ylabel("Largeur du pétale (cm)")
    plt.title("Pétales des iris par espèce")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHIQUES, "nuage_points.png"), dpi=120)
    plt.close()

    corr = df[noms_features].corr()

    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Corrélation")

    etiquettes = [n.replace(" (cm)", "") for n in noms_features]
    plt.xticks(range(4), etiquettes, rotation=45, ha="right")
    plt.yticks(range(4), etiquettes)

    for i in range(4):
        for j in range(4):
            plt.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center"
            )

    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHIQUES, "correlation.png"), dpi=120)
    plt.close()

    print(f"Graphiques enregistrés dans le dossier '{GRAPHIQUES}/'\n")


def entrainer_et_evaluer(X, y, noms_classes):
    """Entraîne et évalue deux modèles de classification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=GRAINE, stratify=y
    )

    print(
        f"Entraînement : {len(X_train)} fleurs | "
        f"Test : {len(X_test)} fleurs\n"
    )

    modeles = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Arbre de décision": DecisionTreeClassifier(random_state=GRAINE),
    }

    resultats = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (nom, modele) in zip(axes, modeles.items()):
        modele.fit(X_train, y_train)
        y_pred = modele.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        resultats[nom] = precision

        print(f"--- {nom} ---")
        print(f"Précision : {precision:.2%}")
        print("Matrice de confusion :")
        print(confusion_matrix(y_test, y_pred), "\n")

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=noms_classes,
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"{nom} ({precision:.0%})")

    plt.tight_layout()
    plt.savefig(
        os.path.join(GRAPHIQUES, "matrices_confusion.png"),
        dpi=120,
    )
    plt.close()

    meilleur = max(resultats, key=resultats.get)
    print(f"Meilleur modèle : {meilleur} ({resultats[meilleur]:.2%})\n")

    return modeles[meilleur]


def predire(modele, noms_classes, noms_features, mesures):
    """Prédit l'espèce d'une fleur à partir de ses mesures."""
    fleur = pd.DataFrame([mesures], columns=noms_features)
    indice = modele.predict(fleur)[0]
    return noms_classes[indice]


def main():
    parser = argparse.ArgumentParser(
        description="Projet ML : fleurs Iris"
    )
    parser.add_argument(
        "--predire",
        nargs=4,
        type=float,
        metavar=(
            "LONG_SEPALE",
            "LARG_SEPALE",
            "LONG_PETALE",
            "LARG_PETALE",
        ),
        help="Mesures d'une fleur en cm pour prédire son espèce",
    )
    args = parser.parse_args()

    iris, df = charger_donnees()
    noms_features = list(iris.feature_names)
    noms_classes = list(iris.target_names)

    explorer(df)
    visualiser(df, noms_features)

    meilleur_modele = entrainer_et_evaluer(
        df[noms_features],
        df["target"],
        noms_classes,
    )

    mesures = args.predire or [5.1, 3.5, 1.4, 0.2]
    espece = predire(
        meilleur_modele,
        noms_classes,
        noms_features,
        mesures,
    )

    print(f"Mesures {mesures} -> espèce prédite : {espece}")


if __name__ == "__main__":
    main()
