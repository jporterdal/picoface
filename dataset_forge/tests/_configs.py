from dataset_forge.config import ForgeConfig
from dataset_forge.export import DEFAULT_CONFIG


def tiny_config(**overrides) -> ForgeConfig:
    """A few images of three classes: fast enough for any test."""
    values = dict(
        name="tiny",
        class_names=("circle", "ring", "square"),
        train_per_class=8,
        test_per_class=4,
    )
    values.update(overrides)
    return ForgeConfig(**values)


def default_config(**overrides) -> ForgeConfig:
    """The checked-in default config, with any fields replaced."""
    values = ForgeConfig.load(DEFAULT_CONFIG).to_dict()
    values.update(overrides)
    return ForgeConfig.from_dict(values)
