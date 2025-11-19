"""LightningCLI entry point for twin training

This provides a config-driven training interface using LightningCLI.
Note: Twin training requires a foundation model to be loaded first.
Example usage:
    python -m src.training.cli_twin fit --config config/twin_config.yaml
"""

from pytorch_lightning.cli import LightningCLI
from src.training.train_twins_lightning import TwinLightningModule
from src.training.data_modules import TwinDataModule


def cli_main():
    """CLI entry point for twin training."""
    cli = LightningCLI(
        TwinLightningModule,
        TwinDataModule,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    cli_main()

