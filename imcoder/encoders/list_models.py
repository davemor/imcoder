import logging

import click

from imcoder.encoders.models import get_availible_models


@click.command()
def list_models():
    logging.info("Welcome to imcoder!")
    logging.info("The following models can be used to encode your images:")
    model_names = get_availible_models()
    for name in model_names:
        print(name)
