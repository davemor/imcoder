import logging
from pathlib import Path

import click
import numpy as np
from scipy.io import savemat
import zarr
from tqdm import tqdm
import multiprocessing as mp
from pprint import pprint
import torch
from torch.utils.data import DataLoader

from imcoder.encoders.loaders import UnlabelledImageFolder
from imcoder.encoders.models import get_model


logging.basicConfig(level=logging.INFO)


def save_features(arr: np.array, path: Path, format: str) -> None:
    if format == "csv":
        np.savetxt(path, arr, delimiter=",")
    elif format == "npy":
        np.save(path, arr)
    elif format == "mat":
        savemat(path, {"features": arr, "label": "embeddings"})
    elif format == "zaar":
        store = zarr.DirectoryStore(path)
        root = zarr.open_group(store=store, mode='w')
        dset = root.create_dataset('features', shape=arr.shape, chunks=arr.shape, dtype=arr.dtype)
        dset[:] = arr


def encode_images(model, preprocess, input_dir: Path, batch_size: int, device: str, num_workers: int) -> np.array:
    dataset = UnlabelledImageFolder(input_dir, preprocess)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers)
    with torch.no_grad():
        features = [model(xs.to(device)).detach().cpu().numpy() for xs in tqdm(loader)]
    return np.concatenate(features)


@click.command()
@click.argument("input_dir", type=click.Path(exists=False))
@click.argument("output_path", type=click.Path(exists=False))
@click.argument("model_name", type=click.STRING)
@click.argument("batch_size", type=click.INT, default=64)
@click.option("--dirs", "-d", is_flag=True, help="Expect a directory of directories.")
@click.option(
    "--format",
    type=click.Choice(["csv", "npy", "mat", "zarr"]),
    default="csv",
    help="output format",
)
def encode(input_dir, output_path, model_name, batch_size, dirs, format):
    logging.info("Welcome to imcoder.")

    Path(output_path).mkdir(parents=True, exist_ok=True)

    # find all the directories to look for images in
    if dirs:
        image_dirs = [f for f in Path(input_dir).iterdir() if f.is_dir()]
        image_dirs = sorted(image_dirs, key=lambda p: p.name)
    else:
        image_dirs = input_dir
    logging.info(f"Found {len(image_dirs)} image directories.")

    # setup the pytorch device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Running on device: {device}")

    # get the model from torchvision
    logging.info(f"Loading {model_name} model.")
    model, preprocess = get_model(model_name)
    model.to(device)
    model.eval()  # set the model into evaluation mode
    logging.info(model)

    # "huristic" to compute the number of loader workers
    num_workers = max(mp.cpu_count() // 2, 1)
    logging.info(f"Loader will use {num_workers} workers.")

    # iterate over the input dirs, encoding and outputting to disk
    for image_dir in image_dirs:
        logging.info(f"Encoding {image_dir}")
        features = encode_images(model, preprocess, image_dir, batch_size, device, num_workers)
        filepath = Path(output_path, image_dir.stem).with_suffix(f".{format}")
        logging.info(f"Saving embeddings to {filepath}.")
        save_features(features, filepath, format)

    logging.info("Complete.")
