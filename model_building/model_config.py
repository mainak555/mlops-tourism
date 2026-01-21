
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
)

from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint, uniform

MODEL_CONFIG = {
    "BaggingClassifier": {
        "estimator": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            random_state=42
        ),
        "grid_params": {
            "baggingclassifier__bootstrap": [True, False],
            "baggingclassifier__n_estimators": randint(50, 200),
            "baggingclassifier__max_samples": uniform(0.5, 0.5),
            "baggingclassifier__max_features": uniform(0.5, 0.5),
            "baggingclassifier__estimator__max_depth": randint(3, 11),
            "baggingclassifier__estimator__min_samples_leaf": randint(1, 11),
            "baggingclassifier__estimator__min_samples_split": randint(2, 11),
            "baggingclassifier__estimator__max_leaf_nodes": randint(5, 20),
            "baggingclassifier__estimator__class_weight": ["balanced", None],
            "baggingclassifier__estimator__criterion": ["gini", "entropy"],
        }
    },
    "AdaBoostClassifier": {
        "estimator": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=42), random_state=42
        ),
        "grid_params": {
            "adaboostclassifier__n_estimators": randint(50, 200),
            "adaboostclassifier__learning_rate": uniform(0.01, 1.0),
            "adaboostclassifier__estimator__max_depth": randint(3, 11),
            "adaboostclassifier__estimator__min_samples_leaf": randint(1, 11),
            "adaboostclassifier__estimator__min_samples_split": randint(2, 11),
            "adaboostclassifier__estimator__max_leaf_nodes": randint(5, 20),
            "adaboostclassifier__estimator__class_weight": [None, "balanced"],
            "adaboostclassifier__estimator__criterion": ["gini", "entropy"],
        }
    }
}
