import pandas as pd
import numpy as np
import sqlite3
from sklearn.decomposition import PCA

from paths import db_path
from sentence_transformers import SentenceTransformer
import torch
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def get_training_data() -> pd.DataFrame:
    print("Fetching training data")
    with sqlite3.connect(str(db_path())) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM training")
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=["id", "a", "b", "c", "score"]).drop(columns=["id"]).dropna()
    df["score"] = df["score"].astype(int)
    return df

def get_dish_data() -> pd.DataFrame:
    print("Fetching dish data")
    with sqlite3.connect(str(db_path())) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dishes")
        rows = cursor.fetchall()

    return pd.DataFrame(rows, columns=["id", "Nimi", "Kuvaus", "Pääraaka-aine", "Raskas", "Hinta", "Toistuvuus"]).drop(columns=["id"]).dropna()


def build_feature_df(df: pd.DataFrame, dishes_df: pd.DataFrame, embedding_map: dict[str, np.array], scores: pd.Series = None) -> pd.DataFrame:
    print("Building feature df")
    assert(list(df.columns) == ["a", "b", "c"])

    df = df.copy()
    original_df = df.copy()

    name_to_id = dict(zip(dishes_df["Nimi"].to_list(), dishes_df.index))
    id_to_name = dishes_df["Nimi"].to_dict()

    for col in ["a", "b", "c"]:
        df[col] = df[col].map(lambda x: name_to_id[x])

    dish_features = pd.get_dummies(dishes_df.drop(columns=["Nimi", "Kuvaus"]),
                    columns=["Pääraaka-aine", "Raskas", "Hinta", "Toistuvuus"])
    dish_features_columns = dish_features.columns.to_list()
    dish_features = dish_features.to_numpy()
    indexes = df.iloc[:,:3].to_numpy()
    X = np.array([dish_features[list(idx_triple)].sum(axis=0) for idx_triple in indexes])

    feature_df = pd.DataFrame(X, columns=dish_features_columns)
    if scores is not None:
        feature_df.insert(len(feature_df.columns), "score", scores)

    feature_df["protein_entropy"] = feature_df.apply(lambda row: -np.array([x/3 * np.log2(x/3) if x > 0 else 0 for x in row[:4]]).sum(), axis=1)
    feature_df["raskas_entropy"] = feature_df.apply(lambda row: -np.array([x/3 * np.log2(x/3) if x > 0 else 0 for x in row[4:7]]).sum(), axis=1)
    feature_df["hinta_entropy"] = feature_df.apply(lambda row: -np.array([x/3 * np.log2(x/3) if x > 0 else 0 for x in row[7:10]]).sum(), axis=1)

    feature_df["raskas_var"] = feature_df.apply(lambda row: np.var(row[4:7]), axis=1)
    feature_df["raskas_mean"] = feature_df.apply(lambda row: np.mean(row[4:7]), axis=1)

    feature_df["raskas_sum"] = feature_df["Raskas_1"] * 1 + feature_df["Raskas_2"] * 2 + feature_df["Raskas_3"] * 3
    feature_df["hinta_sum"] = feature_df["Hinta_1"] * 1 + feature_df["Hinta_2"] * 2 + feature_df["Hinta_3"] * 3
    feature_df["hinta_sum_square"] = feature_df["hinta_sum"] ** 2

    feature_df["toistuvuus_sum"] = feature_df["Toistuvuus_1"] * 10 + feature_df["Toistuvuus_2"] * 5 + feature_df["Toistuvuus_3"] * 1 + feature_df["Toistuvuus_4"] * 0

    feature_df["katkarapu_ja_kala"] = feature_df["Pääraaka-aine_Kala"] + feature_df["Pääraaka-aine_Katkarapu"]

    feature_df["similarity"] = original_df.apply(lambda row: similarity(food_names=list(row[:3]), embedding_map=embedding_map), axis=1)

    return feature_df


def build_embeddings(dish_explanations: dict[str, str]):
    """
    Takes in a dict with pairs dish_name: dish_description
    Returns a dict with pairs dish_name: np.array
    """
    print("Calculating embeddings")
    dish_explanation_values = list(dish_explanations.values())
    embeddings = model.encode(dish_explanation_values)
    embeddings_2d = pca.fit_transform(embeddings)
    return {dish_name: embeddings_2d[i] for i, dish_name in enumerate(dish_explanations.keys())}


def similarity(food_names: list[str], embedding_map: dict[str, str]):
    embeddings_2d = np.array([embedding_map[name] for name in food_names])
    sim = model.similarity(embeddings_2d, embeddings_2d)
    return (1/sim[tuple(torch.triu_indices(len(sim), len(sim), offset=1))].abs()).sum().sqrt().item()


def build_model(feature_df: pd.DataFrame):
    print("Fitting model to data")
    X = feature_df[["protein_entropy", "raskas_entropy", "raskas_sum", "similarity", "toistuvuus_sum", "katkarapu_ja_kala"]].to_numpy()
    y = feature_df["score"].to_numpy()

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    regr = LinearRegression(fit_intercept=False)
    regr.fit(X_poly, y)

    return regr


def predict(dishes_df: pd.DataFrame, model, embedding_map):
    print("Calculating predicions")
    names = np.array(dishes_df["Nimi"].unique())

    comb_idx = list(combinations(range(len(names)), 3))
    result_df = pd.DataFrame()
    result_df.insert(0, "c", [names[c] for (_, _, c) in comb_idx])
    result_df.insert(0, "b", [names[b] for (_, b, _) in comb_idx])
    result_df.insert(0, "a", [names[a] for (a, _, _) in comb_idx])

    result_df_with_features = build_feature_df(result_df, dishes_df, embedding_map)

    X = result_df_with_features[["protein_entropy", "raskas_entropy", "raskas_sum", "similarity", "toistuvuus_sum", "katkarapu_ja_kala"]].to_numpy()
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    result_df["rating"] = model.predict(X_poly)

    with sqlite3.connect(str(db_path())) as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS predicted_scores")

    with sqlite3.connect(str(db_path())) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE predicted_scores(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        a TEXT,
        b TEXT,
        c TEXT,
        rating INT
        )
        """)
        conn.commit()

    with sqlite3.connect(str(db_path())) as conn:
        cursor = conn.cursor()
        for i, row in result_df.iterrows():
            cursor.execute("INSERT INTO predicted_scores (a, b, c, rating) VALUES (?, ?, ?, ?)", row.tolist())
        conn.commit()


print("Loading embedding model")
model = SentenceTransformer("all-MiniLM-L6-v2")
model.similarity_fn_name = "euclidean"
pca = PCA(2)

training_df = get_training_data()
dishes_df = get_dish_data()

embedding_map = build_embeddings(dishes_df[["Nimi", "Kuvaus"]].set_index("Nimi").to_dict()["Kuvaus"])

training_features_df = build_feature_df(training_df[["a", "b", "c"]], dishes_df, embedding_map, training_df["score"])
regressor_model = build_model(training_features_df)

predict(dishes_df, regressor_model, embedding_map)
print("--- Predicted scores are now calculated! ---")