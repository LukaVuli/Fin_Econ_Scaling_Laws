"""Persistence layer for experiment results."""

from __future__ import annotations

import gc
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from .config import OutputConfig, ScalingLawConfig
from .enums import ResumeMode


class ResultsManager:
    """Manager class for saving and loading experiment results."""

    def __init__(self, output_dir: Optional[Union[str, Path]], config: Optional[ScalingLawConfig]):
        """
        Initialize the results manager.

        Args:
            output_dir: Directory for saving results
            config: ScalingLawConfig instance
        """
        self.config = config or ScalingLawConfig(
            output=OutputConfig(output_dir=str(output_dir or "./Output/"))
        )
        self.output_config = self.config.output
        self.artifacts = self.output_config.artifacts

        output_root = output_dir if output_dir is not None else self.output_config.output_dir
        self.output_path = Path(output_root)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.pkl_path = self.artifact_path("results_pickle")
        self.json_path = self.artifact_path("results_json")
        self.csv_path = self.artifact_path("portfolio_returns_csv")
        self.test_sample_path = self.artifact_path("test_sample_csv")
        self.models_dir = self.artifact_path("models_dir")

    def _resolve_artifact_path(self, artifact_name: str) -> Path:
        path = Path(artifact_name)
        if path.is_absolute():
            return path
        return self.output_path / path

    def artifact_path(self, artifact_field: str) -> Path:
        """Return the resolved path for an ArtifactNames field."""
        return self._resolve_artifact_path(getattr(self.artifacts, artifact_field))

    def model_path(self, model_name: str) -> Path:
        """Return the configured save path for a trained Keras model."""
        return self.models_dir / f"{model_name}.keras"

    def _load_results_from_path(
            self,
            path: Path,
            loader: Callable[[Any], Any],
            binary: bool = False
    ) -> List[Dict[str, Any]]:
        if not path.exists() or path.stat().st_size == 0:
            return []
        try:
            with open(path, "rb" if binary else "r") as f:
                results = loader(f)
            return results if isinstance(results, list) else []
        except Exception:
            return []

    def load_existing_results(self) -> List[Dict[str, Any]]:
        """Load existing result metadata from pickle, falling back to JSON."""
        pickle_results = self._load_results_from_path(self.pkl_path, pickle.load, binary=True)
        if pickle_results:
            return pickle_results
        return self._load_results_from_path(self.json_path, json.load)

    def load_existing_model_names(self) -> set:
        """Load model_name values from existing result artifacts."""
        return {
            result["model_name"]
            for result in self.load_existing_results()
            if isinstance(result, dict) and result.get("model_name")
        }

    def has_existing_results(self) -> bool:
        """Return True if any configured result artifact already contains data."""
        if self.load_existing_results():
            return True
        return self.csv_path.exists() and self.csv_path.stat().st_size > 0

    def initialize_files(self, resume: Union[ResumeMode, str]):
        """
        Initialize output files based on resume mode.

        Args:
            resume: ``ResumeMode`` instance or one of its ``.value`` strings.
        """
        resume_mode = ResumeMode.coerce(resume)

        if resume_mode == ResumeMode.OVERWRITE:
            print("✓ Fresh start mode: Resetting output files")
            self.pkl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pkl_path, 'wb') as f:
                pickle.dump([], f)
            if self.csv_path.exists():
                os.remove(self.csv_path)
            if self.json_path.exists():
                os.remove(self.json_path)
            return

        existing_count = len(self.load_existing_results())

        if resume_mode == ResumeMode.FAIL_IF_EXISTS:
            if self.has_existing_results():
                raise FileExistsError(
                    f"Configured results already exist in {self.output_path}. "
                    "Use ResumeMode.UPDATE_EXISTING, ResumeMode.OVERWRITE, "
                    "or ResumeMode.SKIP_EXISTING to continue."
                )
            print("✓ Fail-if-exists mode: No existing results found")
            return

        print(f"✓ Resume mode ({resume_mode.value}): Found {existing_count} existing model(s)")
        if resume_mode == ResumeMode.SKIP_EXISTING:
            print("  Existing model_name entries will be skipped; new models will be added")
        else:
            print(f"  Models will be updated/added as training proceeds")

    def save_test_sample(self, test_sample: pd.DataFrame) -> Path:
        """Save the configured test-sample CSV artifact and return its path."""
        self.test_sample_path.parent.mkdir(parents=True, exist_ok=True)
        test_sample.to_csv(self.test_sample_path, index=False)
        return self.test_sample_path

    def save_result_to_pickle(self, result: Dict[str, Any]):
        """
        Save a single result to the pickle file.

        Args:
            result: Result dictionary to save
        """
        if not self.config.output.save_pickle:
            return

        current_results_list = []

        if self.pkl_path.exists() and self.pkl_path.stat().st_size > 0:
            try:
                with open(self.pkl_path, 'rb') as f:
                    current_results_list = pickle.load(f)
            except Exception as e:
                print(f"⚠ Could not load existing pickle: {e}")
                current_results_list = []

        model_name = result.get('model_name')
        found = False
        for i, existing_result in enumerate(current_results_list):
            if existing_result.get('model_name') == model_name:
                current_results_list[i] = result
                found = True
                print(f"Save Status: Updated existing entry for {model_name}")
                break

        if not found:
            current_results_list.append(result)
            print(f"Save Status: Added new entry for {model_name}")

        self.pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(current_results_list, f)

        del current_results_list
        gc.collect()

    def save_result_to_json(self):
        """Update JSON file from pickle file."""
        if not self.config.output.save_json:
            return

        try:
            with open(self.pkl_path, 'rb') as f:
                results_list = pickle.load(f)

            json_safe_results = []
            for r in results_list:
                r_copy = {k: v for k, v in r.items() if k not in ['decile_returns', 'ts_returns']}
                json_safe_results.append(r_copy)

            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.json_path, 'w') as f:
                json.dump(json_safe_results, f, indent=2)

            del results_list
            del json_safe_results
            gc.collect()

        except Exception as e:
            print(f"⚠ Could not update JSON: {e}")

    @staticmethod
    def _sanitize_returns_index(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a returns panel's date index before an outer-join merge.

        Guards the two ways a returns CSV can silently corrupt across
        ``ResumeMode.UPDATE_EXISTING`` reruns:

        * NaT / blank date labels (e.g. an undated portfolio row written as an
          empty label and read back as NaT). Duplicate NaT labels make an
          outer ``join`` do a cartesian product, so the tail duplicates and
          grows on every rerun.
        * A string / object index (when ``parse_dates`` cannot cleanly parse
          the column) that will not align with a fresh ``DatetimeIndex``.

        Coerces the index to datetime, drops unparseable / NaT rows, and
        collapses duplicate labels (keeping the last) so the join stays 1:1.
        """
        idx = pd.to_datetime(df.index, errors='coerce')
        df = df.set_axis(idx, axis=0)
        df = df[df.index.notna()]
        df = df[~df.index.duplicated(keep='last')]
        df.index.name = 'date'
        return df

    def _save_returns_to_csv(
            self,
            returns_df: pd.DataFrame,
            model_identifier: str
    ):
        """Merge a model's returns columns into the shared returns CSV.

        Shared implementation for the panel (decile) and time-series writers.
        Columns are namespaced by ``model_identifier`` and merged onto any
        existing on-disk panel with a date-aligned outer join. Both sides are
        passed through :meth:`_sanitize_returns_index` first so an
        ``UPDATE_EXISTING`` rerun cannot accumulate a duplicated tail.
        """
        if not self.config.output.save_csv:
            return

        new_data = {}
        for col in returns_df.columns:
            col_name = f"{model_identifier}_{col}"
            new_data[col_name] = returns_df[col].values

        new_df = pd.DataFrame(new_data, index=returns_df.index)
        new_df.index.name = 'date'
        new_df = self._sanitize_returns_index(new_df)

        if self.csv_path.exists():
            try:
                existing_df = pd.read_csv(self.csv_path, index_col='date', parse_dates=True)
                existing_df = self._sanitize_returns_index(existing_df)

                cols_to_remove = [c for c in existing_df.columns
                                  if c.startswith(f"{model_identifier}_")]
                if cols_to_remove:
                    existing_df = existing_df.drop(columns=cols_to_remove)
                    print(f"  ↻ Replacing existing CSV columns for {model_identifier}")

                combined_df = existing_df.join(new_df, how='outer').sort_index()
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                combined_df.to_csv(self.csv_path)

                del existing_df
                del combined_df
            except Exception as e:
                print(f"⚠ Error updating CSV, overwriting: {e}")
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                new_df.to_csv(self.csv_path)
        else:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(self.csv_path)

        del new_df
        gc.collect()

    def save_decile_returns_to_csv(
            self,
            decile_returns: pd.DataFrame,
            model_identifier: str
    ):
        """
        Save decile returns to CSV file.

        Args:
            decile_returns: DataFrame with decile returns
            model_identifier: Identifier for the model
        """
        self._save_returns_to_csv(decile_returns, model_identifier)

    def save_ts_returns_to_csv(
            self,
            ts_returns: pd.DataFrame,
            model_identifier: str
    ):
        """
        Save time-series strategy returns to CSV file.

        Args:
            ts_returns: DataFrame with time-series strategy returns
            model_identifier: Identifier for the model
        """
        self._save_returns_to_csv(ts_returns, model_identifier)

    def load_results(self) -> List[Dict[str, Any]]:
        """Load all results from pickle file."""
        if self.pkl_path.exists():
            with open(self.pkl_path, 'rb') as f:
                return pickle.load(f)
        return []
