import os
import numpy as np
import torch
from .models.diff_model import diff_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import click

# python -m src.infer --loadDir models --loadFile model_235e_460000s.pkl --loadDefFile model_params_235e_460000s.json

# python -m src.infer --loadDir models/models_res_res --loadFile model_438e_550000s.pkl --loadDefFile model_params_438e_550000s.json

@click.command()

# Required
@click.option("--loadDir", "loadDir", type=str, default="models/models_res_res", help="Location of the models to load in.", required=True)
@click.option("--loadFile", "loadFile", type=str, default="model_438e_550000s.pkl", help="Name of the .pkl model file to load in. Ex: model_438e_550000s.pkl", required=True)
@click.option("--loadDefFile", "loadDefFile", type=str, default="model_params_438e_550000s.json", help="Name of the .json model file to load in. Ex: model_params_438e_550000s.json", required=True)

# Generation parameters
@click.option("--num_images", "num_images", type=int, default=200, help="Number of images to generate.", required=False)
@click.option("--step_size", "step_size", type=int, default=10, help="Step size when generating. A step size of 10 with a model trained on 1000 steps takes 100 steps to generate. Lower is faster, but produces lower quality images.", required=False)

@click.option("--DDIM_scale", "DDIM_scale", type=int, default=1, help=" 1:DDPM  0:DDIM.", required=False)

@click.option("--device", "device", type=str, default="gpu", help="Device to put the model on. use \"gpu\" or \"cpu\".", required=False)
@click.option("--guidance", "w", type=int, default=4, help="Classifier guidance scale which must be >= 0. The higher the value, the better the image quality, but the lower the image diversity.", required=False)
@click.option("--class_label", "class_label", type=int, default=3, help="0-indexed class value. Use -1 for a random class and any other class value >= 0 for the other classes. FOr imagenet, the class value range from 0 to 999 and can be found in data/class_information.txt", required=False)
@click.option("--corrected", "corrected", type=bool, default=False, help="True to put a limit on generation, False to not put a litmit on generation. If the model is generating images of a single color, then you may need to set this flag to True. Note: This restriction is usually needed when generating long sequences (low step size) Note: With a higher guidance w, the correction usually messes up generation.", required=False)

# Output parameters
@click.option("--output_dir", "output_dir", type=str, default="output_images/testcase/DDPM", help="Directory to save the output images.", required=False)

def infer(
    loadDir: str,
    loadFile: str,
    loadDefFile: str,
    num_images: int,
    step_size: int,
    DDIM_scale: int,
    device: str,
    w: int,
    class_label: int,
    corrected: bool,
    output_dir: str
    ):

    os.makedirs(output_dir, exist_ok=True)

    ### Model Creation
    # Create a dummy model
    model = diff_model(3, 3, 1, 1, ["res", "res"], 100000, "cosine", 100, device, 100, 1000, 16, 0.0, step_size, DDIM_scale)
    
    # Load in the model weights
    model.loadModel(loadDir, loadFile, loadDefFile)

    for i in range(num_images):
        noise, imgs = model.sample_imgs(1, class_label, w, True, True, True, corrected)
        # Convert the sample image to 0->255
        noise = torch.clamp(noise.cpu().detach().int(), 0, 255)
        for j, img in enumerate(noise):
            # img_path = os.path.join(output_dir, f"generated_image_{i + (class_label * 10 + 1)}.png")
            img_path = os.path.join(output_dir, f"generated_image_{i}.png")
            plt.imsave(img_path, img.permute(1, 2, 0).numpy().astype(np.uint8))
            print(f"Saved image {img_path}")

if __name__ == '__main__':
    infer()

