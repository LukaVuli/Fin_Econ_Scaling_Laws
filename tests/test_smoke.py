"""Smoke tests — confirm the package imports cleanly and configs build."""


def test_package_imports():
    """Top-level package import exposes the public API."""
    import scaling_laws

    assert hasattr(scaling_laws, "ScalingLawConfig")
    assert hasattr(scaling_laws, "ScalingLawExperiment")
    assert hasattr(scaling_laws, "default_epochs_schedule")
    assert hasattr(scaling_laws, "default_taper_schedule")


def test_default_config_constructs():
    """ScalingLawConfig() with no args produces a valid nested config."""
    from scaling_laws import ScalingLawConfig

    config = ScalingLawConfig()
    assert config.architecture.activation == "relu"
    assert config.architecture.output_units == 1
    assert config.training.train_batch_size > 0
    assert config.runtime.vary_seed_per_model is False


def test_default_epochs_schedule():
    """Sanity check the epochs-per-model-size formula."""
    from scaling_laws import default_epochs_schedule

    # 1K params:  0.1 * 1000^0.75  = 17.78 -> 17 + 100 = 117
    assert default_epochs_schedule(1_000) == 117
    # 10K params: 0.1 * 10000^0.75 = 100.00 -> 100 + 100 = 200
    assert default_epochs_schedule(10_000) == 200


def test_nested_config_dict_coercion():
    """ScalingLawConfig accepts plain dicts for nested configs."""
    from scaling_laws import ScalingLawConfig

    config = ScalingLawConfig(training={"learning_rate": 0.01})
    assert config.training.learning_rate == 0.01


def test_fuzzy_stop_fraction_resolves():
    """The new max_extra_fraction translates correctly into an epoch cap."""
    from scaling_laws import FuzzyStopConfig

    resolved = FuzzyStopConfig(enabled=True, max_extra_fraction=0.5).resolve(
        scheduled_epochs=200
    )
    assert resolved.max_extra_epochs == 100
