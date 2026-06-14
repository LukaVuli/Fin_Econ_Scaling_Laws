"""Data-splitting logic for the scaling-law experiment."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PreSplitData, ScalingLawConfig
from .enums import MissingDataPolicy, SplitMode


@dataclass
class DataSplitResult:
    """Materialized train/validation/test split arrays and saveable metadata."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_dates: Optional[np.ndarray]
    test_asset_ids: Optional[np.ndarray] = None
    test_sample: Optional[pd.DataFrame] = None
    train_dates: Optional[List[Any]] = None
    val_dates: Optional[List[Any]] = None
    test_date_values: Optional[List[Any]] = None

    def as_tuple(self) -> Tuple[np.ndarray, ...]:
        return (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
            self.test_dates,
        )


class DataSplitter:
    """Prepare model arrays from DataFrames according to split/missing-data config."""

    def __init__(self, config: ScalingLawConfig):
        self.config = config

    def prepare(
            self,
            df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str = 'xret',
            date_col: str = 'date',
            asset_id_col: Optional[str] = None
    ) -> DataSplitResult:
        """Prepare train/validation/test splits from a DataFrame."""
        mode = self._resolve_mode()
        if mode == SplitMode.PRE_SPLIT:
            return self._prepare_pre_split(self.config.split.pre_split)

        model_cols = self._validate_columns(df, feature_cols, target_col, date_col, asset_id_col)
        required_model_cols = list(dict.fromkeys(list(feature_cols) + [target_col, date_col]))
        position_col = self._internal_position_col(df)
        model_data = df[model_cols].copy()
        model_data[position_col] = np.arange(len(df))
        model_data = self._apply_missing_data(
            model_data,
            required_model_cols,
            feature_cols,
            target_col,
            date_col,
        )
        model_data = model_data.sort_values(date_col)

        if mode == SplitMode.DATE_CUTOFFS:
            train_data, val_data, test_data, split_dates = self._split_by_date_cutoffs(
                model_data, date_col
            )
        elif mode == SplitMode.DATE_PROPORTIONS:
            train_data, val_data, test_data, split_dates = self._split_by_date_proportions(
                model_data, date_col
            )
        elif mode == SplitMode.MASKS:
            train_data, val_data, test_data, split_dates = self._split_by_masks(
                df, model_data, position_col, date_col
            )
        else:
            raise ValueError(f"Unsupported split mode: {mode.value}")

        result = self._build_result(
            original_df=df,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            feature_cols=feature_cols,
            target_col=target_col,
            date_col=date_col,
            asset_id_col=asset_id_col,
            position_col=position_col,
            split_dates=split_dates,
        )

        del train_data, val_data, test_data, model_data
        gc.collect()

        return result

    def _resolve_mode(self) -> SplitMode:
        split_config = self.config.split
        mode = split_config.mode
        if mode != SplitMode.AUTO:
            return mode
        if split_config.pre_split is not None:
            return SplitMode.PRE_SPLIT
        if split_config.has_masks():
            return SplitMode.MASKS
        if isinstance(split_config.test_size, str):
            return SplitMode.DATE_CUTOFFS
        return SplitMode.DATE_PROPORTIONS

    @staticmethod
    def _validate_columns(
            df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str,
            date_col: str,
            asset_id_col: Optional[str] = None
    ) -> List[str]:
        requested_cols = list(feature_cols) + [target_col, date_col]
        if asset_id_col is not None:
            requested_cols.append(asset_id_col)
        missing_cols = [col for col in requested_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                "DataFrame is missing required column(s): "
                f"{', '.join(dict.fromkeys(missing_cols))}"
            )
        return list(dict.fromkeys(requested_cols))

    @staticmethod
    def _internal_position_col(df: pd.DataFrame) -> str:
        base_name = "__scaling_law_original_position__"
        position_col = base_name
        suffix = 1
        while position_col in df.columns:
            position_col = f"{base_name}_{suffix}"
            suffix += 1
        return position_col

    def _apply_missing_data(
            self,
            model_data: pd.DataFrame,
            model_cols: List[str],
            feature_cols: List[str],
            target_col: str,
            date_col: str
    ) -> pd.DataFrame:
        policy = self.config.missing_data.policy
        feature_cols_unique = list(dict.fromkeys(feature_cols))

        if policy == MissingDataPolicy.DROP_ANY:
            return model_data.dropna(subset=model_cols)

        if policy == MissingDataPolicy.DROP_TARGET_ONLY:
            return model_data.dropna(subset=[target_col, date_col])

        if policy == MissingDataPolicy.ERROR:
            missing_counts = model_data[model_cols].isna().sum()
            missing_counts = missing_counts[missing_counts > 0]
            if not missing_counts.empty:
                counts_text = ", ".join(
                    f"{col}={int(count)}" for col, count in missing_counts.items()
                )
                raise ValueError(
                    "MissingDataConfig(policy='error') found missing values in model "
                    f"columns: {counts_text}"
                )
            return model_data

        if policy == MissingDataPolicy.IMPUTE_MEAN:
            cleaned = model_data.dropna(subset=[target_col, date_col]).copy()
            try:
                feature_means = cleaned[feature_cols_unique].mean(numeric_only=False)
            except TypeError as exc:
                raise ValueError(
                    "MissingDataConfig(policy='impute_mean') requires numeric feature "
                    "columns so feature means can be computed"
                ) from exc

            all_missing_features = [
                col for col in feature_cols_unique if pd.isna(feature_means[col])
            ]
            if all_missing_features:
                raise ValueError(
                    "MissingDataConfig(policy='impute_mean') cannot impute feature "
                    "column(s) with all values missing: "
                    f"{', '.join(all_missing_features)}"
                )

            cleaned.loc[:, feature_cols_unique] = cleaned[feature_cols_unique].fillna(feature_means)
            return cleaned

        raise ValueError(f"Unsupported missing-data policy: {policy.value}")

    @staticmethod
    def _unique_dates(model_data: pd.DataFrame, date_col: str) -> List[Any]:
        return sorted(model_data[date_col].unique())

    def _split_by_date_cutoffs(
            self,
            model_data: pd.DataFrame,
            date_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[Any]]]:
        split_config = self.config.split
        try:
            test_cutoff = pd.Timestamp(split_config.test_size)
            val_cutoff = pd.Timestamp(split_config.val_size)
        except Exception as exc:
            raise ValueError(
                "date_cutoffs split mode requires parseable date-like test_size "
                "and val_size values"
            ) from exc

        if val_cutoff >= test_cutoff:
            raise ValueError(
                "date_cutoffs split mode requires val_size cutoff to be before "
                f"test_size cutoff; got val_size={split_config.val_size!r}, "
                f"test_size={split_config.test_size!r}"
            )

        unique_dates = self._unique_dates(model_data, date_col)
        train_dates = [d for d in unique_dates if pd.Timestamp(d) < val_cutoff]
        val_dates = [
            d for d in unique_dates
            if val_cutoff <= pd.Timestamp(d) < test_cutoff
        ]
        test_dates = [d for d in unique_dates if pd.Timestamp(d) >= test_cutoff]

        return self._split_by_date_lists(
            model_data, date_col, train_dates, val_dates, test_dates
        )

    def _split_by_date_proportions(
            self,
            model_data: pd.DataFrame,
            date_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[Any]]]:
        split_config = self.config.split
        try:
            test_size = float(split_config.test_size)
            val_size = float(split_config.val_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "date_proportions split mode requires numeric test_size and val_size"
            ) from exc

        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {split_config.test_size!r}")
        if not 0 < val_size < 1:
            raise ValueError(f"val_size must be between 0 and 1, got {split_config.val_size!r}")

        unique_dates = self._unique_dates(model_data, date_col)
        n_dates = len(unique_dates)
        train_end_idx = int(n_dates * (1 - test_size))
        train_dates = unique_dates[:train_end_idx]
        test_dates = unique_dates[train_end_idx:]

        train_val_split_idx = int(len(train_dates) * (1 - val_size))
        val_dates = train_dates[train_val_split_idx:]
        train_dates = train_dates[:train_val_split_idx]

        return self._split_by_date_lists(
            model_data, date_col, train_dates, val_dates, test_dates
        )

    def _split_by_date_lists(
            self,
            model_data: pd.DataFrame,
            date_col: str,
            train_dates: List[Any],
            val_dates: List[Any],
            test_dates: List[Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[Any]]]:
        train_data = model_data[model_data[date_col].isin(train_dates)]
        val_data = model_data[model_data[date_col].isin(val_dates)]
        test_data = model_data[model_data[date_col].isin(test_dates)]
        return train_data, val_data, test_data, {
            "train": train_dates,
            "val": val_dates,
            "test": test_dates,
        }

    def _split_by_masks(
            self,
            original_df: pd.DataFrame,
            model_data: pd.DataFrame,
            position_col: str,
            date_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[Any]]]:
        split_config = self.config.split
        if not all(
                mask is not None
                for mask in (split_config.train_mask, split_config.val_mask, split_config.test_mask)
        ):
            raise ValueError(
                "masks split mode requires train_mask, val_mask, and test_mask "
                "aligned to the original DataFrame index"
            )

        train_mask = self._mask_to_array(split_config.train_mask, "train_mask", original_df.index)
        val_mask = self._mask_to_array(split_config.val_mask, "val_mask", original_df.index)
        test_mask = self._mask_to_array(split_config.test_mask, "test_mask", original_df.index)

        overlap = (train_mask & val_mask) | (train_mask & test_mask) | (val_mask & test_mask)
        if np.any(overlap):
            raise ValueError(
                f"Split masks overlap on {int(np.sum(overlap)):,} original DataFrame row(s)"
            )

        original_positions = model_data[position_col].to_numpy(dtype=int)
        train_data = model_data.loc[train_mask[original_positions]]
        val_data = model_data.loc[val_mask[original_positions]]
        test_data = model_data.loc[test_mask[original_positions]]

        return train_data, val_data, test_data, {
            "train": self._unique_dates(train_data, date_col),
            "val": self._unique_dates(val_data, date_col),
            "test": self._unique_dates(test_data, date_col),
        }

    @staticmethod
    def _mask_to_array(mask: Any, name: str, df_index: pd.Index) -> np.ndarray:
        if isinstance(mask, pd.Series):
            if mask.index.equals(df_index):
                mask_values = mask.to_numpy()
            else:
                if not df_index.is_unique:
                    raise ValueError(
                        f"{name} must have exactly the original DataFrame index when "
                        "the DataFrame index contains duplicate labels"
                    )
                aligned = mask.reindex(df_index)
                if aligned.isna().any():
                    raise ValueError(
                        f"{name} is missing labels from the original DataFrame index"
                    )
                mask_values = aligned.to_numpy()
        else:
            mask_values = np.asarray(mask)
            if mask_values.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional boolean mask")
            if len(mask_values) != len(df_index):
                raise ValueError(
                    f"{name} length mismatch: expected {len(df_index):,} values aligned "
                    f"to the original DataFrame, got {len(mask_values):,}"
                )

        if len(mask_values) != len(df_index):
            raise ValueError(
                f"{name} length mismatch: expected {len(df_index):,} values aligned "
                f"to the original DataFrame, got {len(mask_values):,}"
            )
        return np.asarray(mask_values, dtype=bool)

    def _build_result(
            self,
            original_df: pd.DataFrame,
            train_data: pd.DataFrame,
            val_data: pd.DataFrame,
            test_data: pd.DataFrame,
            feature_cols: List[str],
            target_col: str,
            date_col: str,
            asset_id_col: Optional[str],
            position_col: str,
            split_dates: Dict[str, List[Any]],
    ) -> DataSplitResult:
        self._validate_non_empty_splits(train_data, val_data, test_data)

        X_train = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data[target_col].values.astype(np.float32)
        X_val = val_data[feature_cols].values.astype(np.float32)
        y_val = val_data[target_col].values.astype(np.float32)
        X_test = test_data[feature_cols].values.astype(np.float32)
        y_test = test_data[target_col].values.astype(np.float32)
        test_dates_array = test_data[date_col].values
        test_asset_ids = (
            test_data[asset_id_col].values
            if asset_id_col is not None
            else None
        )

        save_cols = [
            c for c in [
                date_col,
                target_col,
                asset_id_col,
                'id',
                'permno',
                'market_equity',
                'lme',
                'excntry',
            ]
            if c in original_df.columns
        ]
        save_cols = list(dict.fromkeys(save_cols))
        test_positions = test_data[position_col].to_numpy(dtype=int)
        test_sample = original_df.iloc[test_positions][save_cols].copy()

        result = DataSplitResult(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            test_dates=test_dates_array,
            test_asset_ids=test_asset_ids,
            test_sample=test_sample,
            train_dates=split_dates["train"],
            val_dates=split_dates["val"],
            test_date_values=split_dates["test"],
        )
        self._validate_result_arrays(result)
        return result

    def _prepare_pre_split(
            self,
            pre_split: Optional[PreSplitData]
    ) -> DataSplitResult:
        if pre_split is None:
            raise ValueError("pre_split split mode requires SplitConfig.pre_split")

        result = DataSplitResult(
            X_train=pre_split.X_train,
            y_train=pre_split.y_train,
            X_val=pre_split.X_val,
            y_val=pre_split.y_val,
            X_test=pre_split.X_test,
            y_test=pre_split.y_test,
            test_dates=pre_split.test_dates,
            test_asset_ids=pre_split.test_asset_ids,
            test_sample=pre_split.test_sample,
        )
        self._validate_result_arrays(result)
        return result

    @staticmethod
    def _validate_non_empty_splits(
            train_data: pd.DataFrame,
            val_data: pd.DataFrame,
            test_data: pd.DataFrame
    ):
        empty_splits = [
            split_name
            for split_name, split_data in (
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
            )
            if len(split_data) == 0
        ]
        if empty_splits:
            raise ValueError(
                "Data split produced empty split(s): "
                f"{', '.join(empty_splits)}. Check split_config cutoffs, "
                "proportions, masks, and missing-data policy."
            )

    @staticmethod
    def _validate_result_arrays(result: DataSplitResult):
        split_pairs = (
            ("train", result.X_train, result.y_train),
            ("val", result.X_val, result.y_val),
            ("test", result.X_test, result.y_test),
        )
        for split_name, X, y in split_pairs:
            if len(X) != len(y):
                raise ValueError(
                    f"{split_name} split has mismatched X/y lengths: "
                    f"X_{split_name}={len(X):,}, y_{split_name}={len(y):,}"
                )
            if len(X) == 0:
                raise ValueError(f"{split_name} split is empty")

        if result.test_dates is not None and len(result.test_dates) != len(result.y_test):
            raise ValueError(
                "test_dates length mismatch: "
                f"test_dates={len(result.test_dates):,}, y_test={len(result.y_test):,}"
            )
        if result.test_asset_ids is not None and len(result.test_asset_ids) != len(result.y_test):
            raise ValueError(
                "test_asset_ids length mismatch: "
                f"test_asset_ids={len(result.test_asset_ids):,}, y_test={len(result.y_test):,}"
            )

