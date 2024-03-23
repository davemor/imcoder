import logging

import click
import torchinfo as ti

from imcoder.encoders.models import get_model


@click.command()
@click.argument("model_name")
@click.option("--summary", '-s', is_flag=True, help="Show a summary of the network activations.")
def show(model_name, summary):
    logging.info("Welcome to imcoder!")
    logging.info(f"Showing model: {model_name}")
    model, preprocess = get_model(model_name)
    print()
    print("Image Preprocessor")
    print("------------------")
    print(preprocess)
    print()
    if summary:
        print("Network Summary")
        print("---------------")
        ti.summary(model, input_size=(32, 3, 224, 224))  # torch is channel first be default
    else:
        print("Network Architecture")
        print("--------------------") 
        print(model)
    