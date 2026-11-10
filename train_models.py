#!/usr/bin/env python3
"""
MODULE 2: ML TRAINING & EVALUATION PIPELINE (train_models.py)
Regenerative DC Drive Powertrain
"""

import os
import sys
import time
import pickle
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
        RandomForestClassifier
    )
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_squared_error,
        confusion_matrix,
        classification_report
    )
    SKLEARN_BACKEND = "scikit-learn (Official Native)"
except ImportError:
    SKLEARN_BACKEND = "Pure-NumPy / SciPy Standalone Engine (Drop-In Fallback)"

    class JoblibFallback:
        @staticmethod
        def dump(obj: Any, filename: str) -> None:
            with open(filename, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        @staticmethod
        def load(filename: str) -> Any:
            with open(filename, "rb") as f:
                return pickle.load(f)

    joblib = JoblibFallback()

    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None

        def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "StandardScaler":
            X_arr = np.asarray(X, dtype=np.float64)
            self.mean_ = np.mean(X_arr, axis=0)
            var = np.var(X_arr, axis=0)
            self.scale_ = np.sqrt(np.where(var == 0, 1.0, var))
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X, dtype=np.float64)
            return (X_arr - self.mean_) / self.scale_

        def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
            return self.fit(X, y).transform(X)

    def train_test_split(
        *arrays,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        stratify: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        rng = np.random.default_rng(random_state)
        n_samples = len(arrays[0])

        if stratify is not None:
            strat_arr = np.asarray(stratify)
            train_idx = []
            test_idx = []
            for c in np.unique(strat_arr):
                c_indices = np.where(strat_arr == c)[0]
                rng.shuffle(c_indices)
                n_test_c = int(np.round(test_size * len(c_indices)))
                test_idx.extend(c_indices[:n_test_c])
                train_idx.extend(c_indices[n_test_c:])
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
            rng.shuffle(train_idx)
            rng.shuffle(test_idx)
        else:
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            n_test = int(np.round(test_size * n_samples))
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]

        results = []
        for arr in arrays:
            arr_np = np.asarray(arr)
            results.append(arr_np[train_idx])
            results.append(arr_np[test_idx])
        return results

    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        return float(1.0 - (ss_res / (ss_tot + 1e-12)))

    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[int]] = None) -> np.ndarray:
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_pred, dtype=int)
        if labels is None:
            labels = np.unique(np.concatenate([y_t, y_p]))
        k = len(labels)
        label_to_idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((k, k), dtype=int)
        for t, p in zip(y_t, y_p):
            if t in label_to_idx and p in label_to_idx:
                cm[label_to_idx[t], label_to_idx[p]] += 1
        return cm

    def classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
        digits: int = 2
    ) -> str:
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_pred, dtype=int)
        classes = np.unique(np.concatenate([y_t, y_p]))
        precisions, recalls, f1s, supports = [], [], [], []

        header = f"{'':>25} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n"
        lines = [header]

        for i, c in enumerate(classes):
            tp = np.sum((y_t == c) & (y_p == c))
            fp = np.sum((y_t != c) & (y_p == c))
            fn = np.sum((y_t == c) & (y_p != c))
            sup = np.sum(y_t == c)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            c_name = target_names[i] if target_names and i < len(target_names) else f"Class {c}"
            lines.append(f"{c_name:>25} {prec:>9.{digits}f} {rec:>9.{digits}f} {f1:>9.{digits}f} {sup:>9d}")

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            supports.append(sup)

        acc = np.mean(y_t == y_p)
        lines.append("")
        lines.append(f"{'accuracy':>25} {'':>9} {'':>9} {acc:>9.{digits}f} {len(y_t):>9d}")

        macro_p = np.mean(precisions)
        macro_r = np.mean(recalls)
        macro_f1 = np.mean(f1s)
        lines.append(f"{'macro avg':>25} {macro_p:>9.{digits}f} {macro_r:>9.{digits}f} {macro_f1:>9.{digits}f} {len(y_t):>9d}")

        tot_sup = np.sum(supports)
        w_p = np.average(precisions, weights=supports) if tot_sup > 0 else 0.0
        w_r = np.average(recalls, weights=supports) if tot_sup > 0 else 0.0
        w_f1 = np.average(f1s, weights=supports) if tot_sup > 0 else 0.0
        lines.append(f"{'weighted avg':>25} {w_p:>9.{digits}f} {w_r:>9.{digits}f} {w_f1:>9.{digits}f} {len(y_t):>9d}")

        return "\n".join(lines)

    class FastDecisionTreeRegressor:
        def __init__(self, max_depth: int = 12, min_samples_split: int = 4, max_features: str = "sqrt"):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.max_features = max_features
            self.tree = None

        def fit(self, X_b: np.ndarray, y: np.ndarray, thresholds: List[np.ndarray], rng: np.random.Generator):
            n_samples, n_features = X_b.shape
            self.thresholds = thresholds
            self.feature_importances_ = np.zeros(n_features, dtype=np.float64)

            if self.max_features == "sqrt":
                self.m_try = max(1, int(np.sqrt(n_features)))
            elif self.max_features is None:
                self.m_try = n_features
            else:
                self.m_try = max(1, int(self.max_features))

            idxs = np.arange(n_samples)
            self.tree = self._build(X_b, y, idxs, 0, rng)
            tot = np.sum(self.feature_importances_)
            if tot > 0:
                self.feature_importances_ /= tot

        def _build(self, X_b: np.ndarray, y: np.ndarray, idxs: np.ndarray, depth: int, rng: np.random.Generator):
            n = len(idxs)
            if depth >= self.max_depth or n < self.min_samples_split:
                return {"val": float(np.mean(y[idxs])) if n > 0 else 0.0}

            y_sub = y[idxs]
            tot_sum = np.sum(y_sub)
            mean = tot_sum / n
            var_curr = np.sum((y_sub - mean) ** 2)

            if var_curr <= 1e-7:
                return {"val": float(mean)}

            n_feat = X_b.shape[1]
            feat_subset = rng.choice(n_feat, size=self.m_try, replace=False)

            best_red = -1.0
            best_f = -1
            best_b = -1

            for f in feat_subset:
                col = X_b[idxs, f]
                max_b = int(np.max(col))
                if max_b <= 0:
                    continue

                bin_cnt = np.bincount(col, minlength=max_b + 1)
                bin_sum = np.bincount(col, weights=y_sub, minlength=max_b + 1)

                cum_cnt = np.cumsum(bin_cnt[:-1])
                cum_sum = np.cumsum(bin_sum[:-1])

                nl = cum_cnt
                nr = n - nl
                valid = (nl >= 2) & (nr >= 2)
                if not np.any(valid):
                    continue

                sum_l = cum_sum[valid]
                sum_r = tot_sum - sum_l
                nl_v = nl[valid]
                nr_v = nr[valid]

                reds = (sum_l ** 2 / nl_v) + (sum_r ** 2 / nr_v) - (tot_sum ** 2 / n)
                best_idx_v = np.argmax(reds)
                if reds[best_idx_v] > best_red:
                    best_red = reds[best_idx_v]
                    best_f = f
                    valid_bins = np.where(valid)[0]
                    best_b = valid_bins[best_idx_v]

            if best_f == -1 or best_b == -1:
                return {"val": float(mean)}

            self.feature_importances_[best_f] += best_red
            mask = X_b[idxs, best_f] <= best_b
            left = self._build(X_b, y, idxs[mask], depth + 1, rng)
            right = self._build(X_b, y, idxs[~mask], depth + 1, rng)

            return {
                "feat": best_f,
                "thresh": float(self.thresholds[best_f][best_b]),
                "left": left,
                "right": right
            }

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.array([self._pred_one(x, self.tree) for x in X])

        def _pred_one(self, x: np.ndarray, node: Dict[str, Any]) -> float:
            if "val" in node:
                return node["val"]
            if x[node["feat"]] <= node["thresh"]:
                return self._pred_one(x, node["left"])
            return self._pred_one(x, node["right"])

    class RandomForestRegressor:
        def __init__(self, n_estimators: int = 100, max_depth: int = 12, min_samples_split: int = 4, random_state: int = 42):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.random_state = random_state
            self.trees = []
            self.feature_importances_ = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
            rng = np.random.default_rng(self.random_state)
            X_arr = np.asarray(X, dtype=np.float64)
            y_arr = np.asarray(y, dtype=np.float64)
            n_samples, n_features = X_arr.shape

            n_bins = 32
            self.thresholds = []
            X_b = np.zeros_like(X_arr, dtype=int)
            for f in range(n_features):
                q = np.unique(np.quantile(X_arr[:, f], np.linspace(0, 1, n_bins + 1)[1:-1]))
                self.thresholds.append(q)
                X_b[:, f] = np.digitize(X_arr[:, f], q)

            self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
            self.trees = []

            for i in range(self.n_estimators):
                boot_idx = rng.integers(0, n_samples, size=n_samples)
                tree = FastDecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    max_features="sqrt"
                )
                tree.fit(X_b[boot_idx], y_arr[boot_idx], self.thresholds, rng)
                self.trees.append(tree)
                self.feature_importances_ += tree.feature_importances_

            tot = np.sum(self.feature_importances_)
            if tot > 0:
                self.feature_importances_ /= tot
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X, dtype=np.float64)
            preds = np.zeros(len(X_arr), dtype=np.float64)
            for tree in self.trees:
                preds += tree.predict(X_arr)
            return preds / len(self.trees)

    class GradientBoostingRegressor:
        def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.1, random_state: int = 42):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.random_state = random_state
            self.trees = []
            self.init_val = 0.0
            self.feature_importances_ = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
            rng = np.random.default_rng(self.random_state)
            X_arr = np.asarray(X, dtype=np.float64)
            y_arr = np.asarray(y, dtype=np.float64)
            n_samples, n_features = X_arr.shape

            self.init_val = float(np.mean(y_arr))
            curr_pred = np.full(n_samples, self.init_val, dtype=np.float64)

            n_bins = 32
            self.thresholds = []
            X_b = np.zeros_like(X_arr, dtype=int)
            for f in range(n_features):
                q = np.unique(np.quantile(X_arr[:, f], np.linspace(0, 1, n_bins + 1)[1:-1]))
                self.thresholds.append(q)
                X_b[:, f] = np.digitize(X_arr[:, f], q)

            self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
            self.trees = []

            for i in range(self.n_estimators):
                residuals = y_arr - curr_pred
                tree = FastDecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=4,
                    max_features=None
                )
                tree.fit(X_b, residuals, self.thresholds, rng)
                curr_pred += self.learning_rate * tree.predict(X_arr)
                self.trees.append(tree)
                self.feature_importances_ += tree.feature_importances_

            tot = np.sum(self.feature_importances_)
            if tot > 0:
                self.feature_importances_ /= tot
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X, dtype=np.float64)
            preds = np.full(len(X_arr), self.init_val, dtype=np.float64)
            for tree in self.trees:
                preds += self.learning_rate * tree.predict(X_arr)
            return preds

    class FastDecisionTreeClassifier:
        def __init__(self, max_depth: int = 12, min_samples_split: int = 4, max_features: str = "sqrt"):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.max_features = max_features
            self.tree = None

        def fit(self, X_b: np.ndarray, y: np.ndarray, n_classes: int, thresholds: List[np.ndarray], rng: np.random.Generator):
            n_samples, n_features = X_b.shape
            self.n_classes = n_classes
            self.thresholds = thresholds
            self.feature_importances_ = np.zeros(n_features, dtype=np.float64)

            if self.max_features == "sqrt":
                self.m_try = max(1, int(np.sqrt(n_features)))
            elif self.max_features is None:
                self.m_try = n_features
            else:
                self.m_try = max(1, int(self.max_features))

            idxs = np.arange(n_samples)
            self.tree = self._build(X_b, y, idxs, 0, rng)
            tot = np.sum(self.feature_importances_)
            if tot > 0:
                self.feature_importances_ /= tot

        def _build(self, X_b: np.ndarray, y: np.ndarray, idxs: np.ndarray, depth: int, rng: np.random.Generator):
            n = len(idxs)
            if n == 0:
                return {"prob": np.zeros(self.n_classes)}

            y_sub = y[idxs]
            counts = np.bincount(y_sub, minlength=self.n_classes)
            prob = counts / n

            if depth >= self.max_depth or n < self.min_samples_split or np.max(prob) >= 0.999:
                return {"prob": prob}

            gini_parent = 1.0 - np.sum(prob ** 2)
            n_feat = X_b.shape[1]
            feat_subset = rng.choice(n_feat, size=self.m_try, replace=False)

            best_gain = -1.0
            best_f = -1
            best_b = -1

            for f in feat_subset:
                col = X_b[idxs, f]
                max_b = int(np.max(col))
                if max_b <= 0:
                    continue

                bin_counts = np.zeros((max_b + 1, self.n_classes), dtype=int)
                for c in range(self.n_classes):
                    bin_counts[:, c] = np.bincount(col[y_sub == c], minlength=max_b + 1)

                cum_counts = np.cumsum(bin_counts[:-1], axis=0)
                nl = np.sum(cum_counts, axis=1)
                nr = n - nl
                valid = (nl >= 2) & (nr >= 2)
                if not np.any(valid):
                    continue

                for b in np.where(valid)[0]:
                    l_cnt = cum_counts[b]
                    r_cnt = counts - l_cnt

                    gini_l = 1.0 - np.sum((l_cnt / nl[b]) ** 2)
                    gini_r = 1.0 - np.sum((r_cnt / nr[b]) ** 2)

                    gain = gini_parent - ((nl[b] / n) * gini_l + (nr[b] / n) * gini_r)
                    if gain > best_gain:
                        best_gain = gain
                        best_f = f
                        best_b = b

            if best_f == -1 or best_b == -1:
                return {"prob": prob}

            self.feature_importances_[best_f] += best_gain * n
            mask = X_b[idxs, best_f] <= best_b
            left = self._build(X_b, y, idxs[mask], depth + 1, rng)
            right = self._build(X_b, y, idxs[~mask], depth + 1, rng)

            return {
                "feat": best_f,
                "thresh": float(self.thresholds[best_f][best_b]),
                "left": left,
                "right": right
            }

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.array([self._pred_one(x, self.tree) for x in X])

        def _pred_one(self, x: np.ndarray, node: Dict[str, Any]) -> np.ndarray:
            if "prob" in node:
                return node["prob"]
            if x[node["feat"]] <= node["thresh"]:
                return self._pred_one(x, node["left"])
            return self._pred_one(x, node["right"])

    class RandomForestClassifier:
        def __init__(self, n_estimators: int = 100, max_depth: int = 12, min_samples_split: int = 4, random_state: int = 42):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.random_state = random_state
            self.trees = []
            self.classes_ = None
            self.feature_importances_ = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
            rng = np.random.default_rng(self.random_state)
            X_arr = np.asarray(X, dtype=np.float64)
            y_arr = np.asarray(y, dtype=int)
            self.classes_ = np.unique(y_arr)
            n_samples, n_features = X_arr.shape
            n_classes = len(self.classes_)

            n_bins = 32
            self.thresholds = []
            X_b = np.zeros_like(X_arr, dtype=int)
            for f in range(n_features):
                q = np.unique(np.quantile(X_arr[:, f], np.linspace(0, 1, n_bins + 1)[1:-1]))
                self.thresholds.append(q)
                X_b[:, f] = np.digitize(X_arr[:, f], q)

            self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
            self.trees = []

            for i in range(self.n_estimators):
                boot_idx = rng.integers(0, n_samples, size=n_samples)
                tree = FastDecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    max_features="sqrt"
                )
                tree.fit(X_b[boot_idx], y_arr[boot_idx], n_classes, self.thresholds, rng)
                self.trees.append(tree)
                self.feature_importances_ += tree.feature_importances_

            tot = np.sum(self.feature_importances_)
            if tot > 0:
                self.feature_importances_ /= tot
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X, dtype=np.float64)
            probs = np.zeros((len(X_arr), len(self.classes_)), dtype=np.float64)
            for tree in self.trees:
                probs += tree.predict_proba(X_arr)
            return probs / len(self.trees)

        def predict(self, X: np.ndarray) -> np.ndarray:
            probs = self.predict_proba(X)
            return self.classes_[np.argmax(probs, axis=1)]


def run_pipeline(
    dataset_path: str = "motor_regen_dataset.csv",
    export_dir: str = ".",
    seed: int = 42
) -> None:
    print("=" * 80)
    print("MODULE 2: ML TRAINING & EVALUATION PIPELINE FOR REGENERATIVE DC DRIVE")
    print("=" * 80)
    print(f"Backend Engine : {SKLEARN_BACKEND}")
    print(f"Dataset Source : {dataset_path}")
    print(f"Artifact Dest  : {os.path.abspath(export_dir)}")
    print("-" * 80)

    if not os.path.exists(dataset_path):
        alt_path = os.path.join("artifacts/file_generation/ttl=63d/output", dataset_path)
        if os.path.exists(alt_path):
            dataset_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find dataset at '{dataset_path}' or '{alt_path}'.")

    df = pd.read_csv(dataset_path)
    print(f"[*] Successfully loaded dataset: {len(df):,} total records, {df.shape[1]} columns.")

    feature_cols = [
        "initial_rpm",
        "pwm_brake_duty",
        "bus_voltage_init",
        "motor_temp_c",
        "peak_current_a",
        "peak_kickback_voltage_v",
        "mean_current_ripple"
    ]

    X = df[feature_cols].values
    y_energy = df["energy_recovered_joules"].values
    y_stopping = df["stopping_time_ms"].values
    y_safety = df["safety_class"].values.astype(int)

    print("[*] Performing 80% Train / 20% Test stratified split across safety classes...")
    (
        X_train_raw, X_test_raw,
        y_energy_train, y_energy_test,
        y_stop_train, y_stop_test,
        y_safety_train, y_safety_test
    ) = train_test_split(
        X, y_energy, y_stopping, y_safety,
        test_size=0.20,
        random_state=seed,
        stratify=y_safety
    )

    print(f"    -> Training set size : {len(X_train_raw):,} samples (80.0%)")
    print(f"    -> Testing set size  : {len(X_test_raw):,} samples (20.0%)")

    print("[*] Normalizing telemetry feature matrix using StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    os.makedirs(export_dir, exist_ok=True)
    scaler_path = os.path.join(export_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"    [+] Saved scaler to: {scaler_path}")

    # MODEL 1: ENERGY RECOVERY REGRESSOR
    print("\n" + "=" * 80)
    print("MODEL 1: ENERGY RECOVERY REGRESSOR (Random Forest)")
    print("=" * 80)
    print("Architecture: Random Forest Regressor (n_estimators=100, max_depth=12)")
    print("Target      : energy_recovered_joules [J]")

    t0 = time.time()
    energy_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=seed
    )
    energy_model.fit(X_train, y_energy_train)
    t_train_energy = time.time() - t0

    y_pred_energy = energy_model.predict(X_test)
    r2_energy = r2_score(y_energy_test, y_pred_energy)
    mae_energy = mean_absolute_error(y_energy_test, y_pred_energy)
    rmse_energy = np.sqrt(mean_squared_error(y_energy_test, y_pred_energy))

    print(f"[+] Model Training Time : {t_train_energy:.2f} s")
    print(f"[*] Evaluation Metrics (Test Set, N = {len(X_test):,}):")
    print(f"    * Coefficient of Determination (R²) : {r2_energy:8.4f}")
    print(f"    * Mean Absolute Error (MAE)         : {mae_energy:8.4f} J")
    print(f"    * Root Mean Squared Error (RMSE)    : {rmse_energy:8.4f} J")

    print("\n[*] Feature Importances (Energy Recovery):")
    imp_energy = energy_model.feature_importances_
    sorted_idx_e = np.argsort(imp_energy)[::-1]
    for rank, idx in enumerate(sorted_idx_e, 1):
        bar = "█" * int(imp_energy[idx] * 40)
        print(f"    {rank}. {feature_cols[idx]:<25} : {imp_energy[idx]:6.4f} ({imp_energy[idx]*100:5.2f}%)  {bar}")

    energy_model_path = os.path.join(export_dir, "energy_model.joblib")
    joblib.dump(energy_model, energy_model_path)
    print(f"\n[+] Saved Energy Recovery Regressor to: {energy_model_path}")

    # MODEL 2: STOPPING TIME REGRESSOR
    print("\n" + "=" * 80)
    print("MODEL 2: STOPPING TIME REGRESSOR (Gradient Boosting)")
    print("=" * 80)
    print("Architecture: Gradient Boosting Regressor (n_estimators=100, max_depth=4, lr=0.1)")
    print("Target      : stopping_time_ms [ms]")

    t0 = time.time()
    stopping_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed
    )
    stopping_model.fit(X_train, y_stop_train)
    t_train_stop = time.time() - t0

    y_pred_stop = stopping_model.predict(X_test)
    r2_stop = r2_score(y_stop_test, y_pred_stop)
    mae_stop = mean_absolute_error(y_stop_test, y_pred_stop)
    rmse_stop = np.sqrt(mean_squared_error(y_stop_test, y_pred_stop))

    print(f"[+] Model Training Time : {t_train_stop:.2f} s")
    print(f"[*] Evaluation Metrics (Test Set, N = {len(X_test):,}):")
    print(f"    * Coefficient of Determination (R²) : {r2_stop:8.4f}")
    print(f"    * Mean Absolute Error (MAE)         : {mae_stop:8.4f} ms")
    print(f"    * Root Mean Squared Error (RMSE)    : {rmse_stop:8.4f} ms")

    print("\n[*] Feature Importances (Stopping Time):")
    imp_stop = stopping_model.feature_importances_
    sorted_idx_s = np.argsort(imp_stop)[::-1]
    for rank, idx in enumerate(sorted_idx_s, 1):
        bar = "█" * int(imp_stop[idx] * 40)
        print(f"    {rank}. {feature_cols[idx]:<25} : {imp_stop[idx]:6.4f} ({imp_stop[idx]*100:5.2f}%)  {bar}")

    stopping_model_path = os.path.join(export_dir, "stopping_model.joblib")
    joblib.dump(stopping_model, stopping_model_path)
    print(f"\n[+] Saved Stopping Time Regressor to: {stopping_model_path}")

    # MODEL 3: OPERATING SAFETY CLASSIFIER
    print("\n" + "=" * 80)
    print("MODEL 3: OPERATING SAFETY CLASSIFIER (Random Forest Classifier)")
    print("=" * 80)
    print("Architecture: Random Forest Classifier (n_estimators=100, max_depth=12)")
    print("Target      : safety_class [0 = Inefficient, 1 = Optimal Safe, 2 = Hazard]")

    t0 = time.time()
    safety_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=seed
    )
    safety_classifier.fit(X_train, y_safety_train)
    t_train_cls = time.time() - t0

    y_pred_safety = safety_classifier.predict(X_test)
    class_labels = [0, 1, 2]
    class_names = [
        "Class 0 (Inefficient)",
        "Class 1 (Optimal Safe)",
        "Class 2 (Hazard)"
    ]

    cm = confusion_matrix(y_safety_test, y_pred_safety, labels=class_labels)
    clf_report = classification_report(y_safety_test, y_pred_safety, target_names=class_names, digits=4)

    print(f"[+] Model Training Time : {t_train_cls:.2f} s")
    print("\n[*] Confusion Matrix (Rows = Actual True, Columns = Predicted):")
    header_cm = f"{'':>25} | {'Pred Class 0':>12} | {'Pred Class 1':>12} | {'Pred Class 2':>12} | {'Total':>7}"
    print(header_cm)
    print("-" * len(header_cm))
    for i, name in enumerate(class_names):
        row_str = f"{name:>25} | {cm[i, 0]:>12d} | {cm[i, 1]:>12d} | {cm[i, 2]:>12d} | {np.sum(cm[i, :]):>7d}"
        print(row_str)
    print("-" * len(header_cm))
    print(f"{'Total Predicted':>25} | {np.sum(cm[:, 0]):>12d} | {np.sum(cm[:, 1]):>12d} | {np.sum(cm[:, 2]):>12d} | {len(y_safety_test):>7d}")

    print("\n[*] Classification Performance Report:")
    print(clf_report)

    print("\n[*] Feature Importances (Safety Classification):")
    imp_cls = safety_classifier.feature_importances_
    sorted_idx_c = np.argsort(imp_cls)[::-1]
    for rank, idx in enumerate(sorted_idx_c, 1):
        bar = "█" * int(imp_cls[idx] * 40)
        print(f"    {rank}. {feature_cols[idx]:<25} : {imp_cls[idx]:6.4f} ({imp_cls[idx]*100:5.2f}%)  {bar}")

    classifier_path = os.path.join(export_dir, "safety_classifier.joblib")
    joblib.dump(safety_classifier, classifier_path)
    print(f"\n[+] Saved Operating Safety Classifier to: {classifier_path}")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE: PERSISTED ARTIFACT MANIFEST")
    print("=" * 80)
    artifacts = [
        ("Feature Scaler", "scaler.joblib", scaler_path),
        ("Energy Recovery Regressor", "energy_model.joblib", energy_model_path),
        ("Stopping Time Regressor", "stopping_model.joblib", stopping_model_path),
        ("Operating Safety Classifier", "safety_classifier.joblib", classifier_path)
    ]
    for title, fname, fpath in artifacts:
        sz_kb = os.path.getsize(fpath) / 1024.0 if os.path.exists(fpath) else 0.0
        print(f"  * {title:<30} -> {fname:<25} ({sz_kb:8.2f} KB) at {fpath}")

    alt_dir = "artifacts/file_generation/ttl=63d/output"
    if os.path.exists(alt_dir) and os.path.abspath(export_dir) != os.path.abspath(alt_dir):
        print("\n[*] Synchronizing copies to deliverable output directory...")
        for _, fname, src in artifacts:
            dst = os.path.join(alt_dir, fname)
            with open(src, "rb") as f_in, open(dst, "wb") as f_out:
                f_out.write(f_in.read())
            print(f"    -> Synced {fname} to {dst}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ML models for Regenerative DC Drive Powertrain")
    parser.add_argument("--dataset", type=str, default="motor_regen_dataset.csv", help="Path to telemetry CSV dataset")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save trained model artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split and tree initialization")
    args = parser.parse_args()

    run_pipeline(
        dataset_path=args.dataset,
        export_dir=args.output_dir,
        seed=args.seed
    )