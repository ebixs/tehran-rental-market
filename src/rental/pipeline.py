from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

from .config import RANDOM_STATE
from .features import add_features, build_preprocessor


def make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("prep", build_preprocessor()),
        ("model", model),
    ])


def candidate_models() -> dict:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, n_jobs=-1, random_state=RANDOM_STATE),
    }
