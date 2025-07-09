import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


def tune_knn():
    k_range = list(range(1, 30))
    train_scores_knn = []
    test_scores_knn = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_scores_knn.append(knn.score(X_train, y_train))
        test_scores_knn.append(knn.score(X_test, y_test))

    plt.figure(figsize=(10, 4))
    plt.plot(k_range, train_scores_knn, label="Train Accuracy", marker="o")
    plt.plot(k_range, test_scores_knn, label="Test Accuracy", marker="o")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs K")
    plt.legend()
    plt.grid(True)
    plt.show()


def tune_rf():
    param_grid_rf = {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 5, 10, 20]}

    rf = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(
        rf, param_grid_rf, cv=5, scoring="accuracy", return_train_score=True
    )
    grid_search_rf.fit(X_train, y_train)
    print("Best parameters:", grid_search_rf.best_params_)

    # print(dir(grid_search_rf))
    best_rf = grid_search_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred_rf))
