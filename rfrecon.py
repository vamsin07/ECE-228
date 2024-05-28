import click
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from static import RFStaticDataset
from transformer import TransformerModel, train_model

VALID_MODEL_TYPES = ["transformer", "vqvae"]

@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "-t",
    "--model-type",
    type=click.Choice(VALID_MODEL_TYPES),
    default="transformer",
    help="Type of model to train",
)
def train(
    model_type: str,
):
    num_symbols = 10
    num_samples = 100
    split_ratio = 0.8
    window_size = 1
    train_ds = RFStaticDataset(
        n_symbols=num_symbols,
        n_trials=int(num_samples*split_ratio),
        window=True,
        window_size=window_size,
    )
    test_ds = RFStaticDataset(
        n_symbols=num_symbols,
        n_trials=int(num_samples*(1-split_ratio)),
        window=True,
        window_size=window_size,
    )
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=2, shuffle=False)

    _, _, rxiq_sample = train_ds[0]
    num_windows, _ = rxiq_sample.shape
    model = get_model(model_type, num_windows=num_windows, num_symbols=num_symbols, window_size=window_size)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    model = train_model(
        model,
        train_dl,
        test_dl,
        criterion,
        optimizer,
        scheduler,
        num_epochs=10,
        train_nbits=int(train_ds.tx.symbol_encoder.get_bps() * train_ds.n_symbols),
        val_nbits=int(test_ds.tx.symbol_encoder.get_bps() * test_ds.n_symbols),
        train_ntrials=train_ds.n_trials,
        val_ntrials=test_ds.n_trials,
    )


def get_model(
    model_type: str,
    num_windows: int,
    num_symbols: int = 10,
    window_size: int = 10,
):
    if model_type == "transformer":
        input_dim = 2 * window_size
        return TransformerModel(
            input_dim=input_dim,
            num_windows=num_windows,
            embedding_dim=8,
            output_dim=num_symbols,
        )
    raise ValueError(f"Invalid model type: {model_type}")


if __name__ == "__main__":
    cli()