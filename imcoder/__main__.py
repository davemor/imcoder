import logging
import click
import multiprocessing as mp

from imcoder.encoders.encode import encode
from imcoder.encoders.show import show
from imcoder.encoders.list_models import list_models


# let's output the info
logging.basicConfig(level=logging.INFO, format='%(message)')


@click.group(help="CLI tool to encode images for similarity search.")
def cli():
    pass


cli.add_command(encode)
cli.add_command(show)
cli.add_command(list_models)


if __name__ == "__main__":
    # mp.set_start_method("fork")
    cli(prog_name="imcoder")
