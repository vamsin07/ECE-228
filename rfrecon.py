import click
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from static import RFStaticDataset
from transformer import TransformerModel, train_model, plot_transformer_losses

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
    window_size = 10
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
    train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=10, shuffle=False)

    _, _, _, rxiq_sample, txiq = train_ds[0]
    num_windows, _ = rxiq_sample.shape
    _, output_dim = txiq.shape
    model = get_model(model_type, num_windows, output_dim, window_size=window_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    _, best_val_loss, train_losses, val_losses, train_bers, val_bers  = train_model(
        model,
        train_dl,
        test_dl,
        criterion,
        optimizer,
        num_epochs=100,
        train_nbits=int(train_ds.tx.symbol_encoder.get_bps() * train_ds.n_symbols),
        val_nbits=int(test_ds.tx.symbol_encoder.get_bps() * test_ds.n_symbols),
        train_ntrials=train_ds.n_trials,
        val_ntrials=test_ds.n_trials,
    )
    plot_transformer_losses(
        train_losses,
        val_losses,
        train_bers,
        val_bers,
    )
    print("Best validation loss:", best_val_loss) 


def get_model(
    model_type: str,
    num_windows: int,
    output_dim: int,
    window_size: int = 10,
):
    if model_type == "transformer":
        input_dim = 2 * window_size
        return TransformerModel(
            input_dim=input_dim,
            num_windows=num_windows,
            embedding_dim=8,
            output_dim=output_dim,
        )
    raise ValueError(f"Invalid model type: {model_type}")


if __name__ == "__main__":
    cli()