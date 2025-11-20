"""LightningCLI entry point for foundation training

This provides a config-driven training interface using LightningCLI.
Example usage:
    python -m src.training.cli_foundation fit --config config/foundation_config.yaml
"""

from pytorch_lightning.cli import LightningCLI
from src.training.train_foundation import FoundationTrainingModule
from src.training.data_modules import FoundationDataModule


def cli_main():
    """CLI entry point for foundation training."""
    cli = LightningCLI(
        FoundationTrainingModule,
        FoundationDataModule,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    cli_main()

