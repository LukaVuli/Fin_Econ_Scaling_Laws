"""Neural-network model builder used by the scaling-law experiment."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    Normalization,
)
from tensorflow.keras.models import Model

from .config import ScalingLawConfig
from .enums import ArchitectureMode, InitializerType, NormalizationType


class ModelBuilder:
    """
    Builder class for creating neural network models with target parameter counts.

    Supports various normalization strategies and architecture modes.
    """

    def __init__(self, config: ScalingLawConfig):
        """
        Initialize the model builder.

        Args:
            config: ScalingLawConfig instance with model configuration
        """
        self.config = config

    def _get_initializer(self, seed: int):
        """Get the appropriate weight initializer."""
        if self.config.architecture.initializer == InitializerType.HE_NORMAL:
            return HeNormal(seed=seed)
        elif self.config.architecture.initializer == InitializerType.GLOROT_UNIFORM:
            return GlorotUniform(seed=seed)
        else:
            return HeNormal(seed=seed)

    def _get_layer_seed(self, layer_index: int) -> int:
        """Get deterministic per-layer seeds from the configured base seed."""
        return int(self.config.runtime.random_state) + layer_index

    def _get_output_initializer(self, layer_count: int):
        """Use Keras' default output initializer, but with an explicit seed."""
        return GlorotUniform(seed=self._get_layer_seed(layer_count))

    def _add_normalization(self, x, layer_index: int):
        """Add normalization layer based on configuration."""
        if self.config.architecture.normalization == NormalizationType.LAYER:
            return LayerNormalization()(x)
        elif self.config.architecture.normalization == NormalizationType.BATCH:
            return BatchNormalization()(x)
        else:  # NormalizationType.NONE
            return x

    def _should_apply_dropout(self, layer_index: int, total_layers: int) -> bool:
        """Determine if dropout should be applied at this layer."""
        if self.config.architecture.dropout_rate <= 0:
            return False
        if self.config.architecture.dropout_middle_only:
            return layer_index > 0 and layer_index < total_layers - 1
        return True

    def _get_tapered_architecture_template(self, target_params: int) -> List[float]:
        """Get the architecture template for tapered mode based on model size."""
        return self.config.architecture.taper_schedule(target_params)

    def build_tapered_model(
            self,
            input_dim: int,
            target_params: int
    ) -> Tuple[Model, Normalization, int, List[int]]:
        """
        Build a neural network with tapered architecture targeting N parameters.

        Args:
            input_dim: Number of input features
            target_params: Target number of parameters

        Returns:
            Tuple of (model, normalizer, actual_params, architecture)
        """
        layers_template = self._get_tapered_architecture_template(target_params)

        def count_params(width):
            total = input_dim * width
            for i in range(len(layers_template) - 1):
                curr_width = int(width * layers_template[i])
                next_width = int(width * layers_template[i + 1])
                total += curr_width * next_width
            total += int(width * layers_template[-1]) * 1
            return total

        # Binary search for optimal width
        low = 1
        high = max(8, int(np.ceil(np.sqrt(target_params))) * 4)
        best_width = 1
        while low <= high:
            mid = (low + high) // 2
            params = count_params(mid)
            if abs(params - target_params) < abs(count_params(best_width) - target_params):
                best_width = mid
            if params < target_params:
                low = mid + 1
            else:
                high = mid - 1

        if abs(count_params(best_width) - target_params) > target_params * 2:
            raise RuntimeError(
                f"Tapered architecture search could not reach target_params={target_params}; "
                f"got {count_params(best_width)} params with width={best_width}. "
                f"Adjust taper_schedule or check binary-search bounds."
            )

        layers_config = [max(1, int(best_width * scale)) for scale in layers_template]

        # Build model
        inputs = Input(shape=(input_dim,))

        if self.config.architecture.use_input_normalization:
            normalizer = Normalization(axis=-1)
            x = normalizer(inputs)
        else:
            normalizer = None
            x = inputs

        for i, neurons in enumerate(layers_config):
            layer_seed = self._get_layer_seed(i)
            kernel_init = self._get_initializer(seed=layer_seed)
            x = Dense(neurons, activation=self.config.architecture.activation, kernel_initializer=kernel_init)(x)
            x = self._add_normalization(x, i)

            if self._should_apply_dropout(i, len(layers_config)):
                x = Dropout(self.config.architecture.dropout_rate, seed=layer_seed)(x)

        outputs = Dense(
            self.config.architecture.output_units,
            activation=self.config.architecture.output_activation,
            kernel_initializer=self._get_output_initializer(len(layers_config)),
        )(x)
        model = Model(inputs=inputs, outputs=outputs)

        actual_params = model.count_params()

        return model, normalizer, actual_params, layers_config

    def build_fixed_depth_model(
            self,
            input_dim: int,
            target_params: int
    ) -> Tuple[Model, Normalization, int, List[int]]:
        """
        Build a neural network with fixed depth and uniform width.

        Args:
            input_dim: Number of input features
            target_params: Target number of parameters

        Returns:
            Tuple of (model, normalizer, actual_params, architecture)
        """
        n_layers = self.config.architecture.fixed_depth_layers

        def count_params(width, n_layers):
            if n_layers == 0:
                return input_dim * 1
            total = input_dim * width
            total += (n_layers - 1) * width * width
            total += width * 1
            return total

        def solve_width(target_params, n_layers):
            if n_layers == 0:
                return 0
            elif n_layers == 1:
                return max(1, int(target_params / (input_dim + 1)))
            else:
                a = n_layers - 1
                b = input_dim + 1
                c = -target_params
                discriminant = b ** 2 - 4 * a * c
                w = (-b + discriminant ** 0.5) / (2 * a)
                return max(1, int(w))

        # Calculate optimal width
        width = solve_width(target_params, n_layers)

        # Fine-tune with binary search
        low, high = max(1, width - 50), width + 50
        best_width = width
        best_diff = abs(count_params(width, n_layers) - target_params)

        while low <= high:
            mid = (low + high) // 2
            params = count_params(mid, n_layers)
            diff = abs(params - target_params)

            if diff < best_diff:
                best_diff = diff
                best_width = mid

            if params < target_params:
                low = mid + 1
            else:
                high = mid - 1

        width = best_width

        # Build model
        inputs = Input(shape=(input_dim,))

        if self.config.architecture.use_input_normalization:
            normalizer = Normalization(axis=-1)
            x = normalizer(inputs)
        else:
            normalizer = None
            x = inputs

        for i in range(n_layers):
            layer_seed = self._get_layer_seed(i)
            kernel_init = self._get_initializer(seed=layer_seed)
            x = Dense(width, activation=self.config.architecture.activation, kernel_initializer=kernel_init)(x)
            x = self._add_normalization(x, i)

            if self._should_apply_dropout(i, n_layers):
                x = Dropout(self.config.architecture.dropout_rate, seed=layer_seed)(x)

        outputs = Dense(
            self.config.architecture.output_units,
            activation=self.config.architecture.output_activation,
            kernel_initializer=self._get_output_initializer(n_layers),
        )(x)
        model = Model(inputs=inputs, outputs=outputs)

        actual_params = model.count_params()
        layers_config = [width] * n_layers

        return model, normalizer, actual_params, layers_config

    def build_model(
            self,
            input_dim: int,
            target_params: int
    ) -> Tuple[Model, Normalization, int, List[int]]:
        """
        Build a model based on the configured architecture mode.

        Args:
            input_dim: Number of input features
            target_params: Target number of parameters

        Returns:
            Tuple of (model, normalizer, actual_params, architecture)
        """
        if self.config.architecture.architecture_mode == ArchitectureMode.TAPERED:
            return self.build_tapered_model(input_dim, target_params)
        else:
            return self.build_fixed_depth_model(input_dim, target_params)
