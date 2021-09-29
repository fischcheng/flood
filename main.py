import sys
import os
from pathlib import Path

from loguru import logger
import numpy as np
import typer
from tifffile import imwrite
from tqdm.auto import tqdm

import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader
from flash.image import SemanticSegmentation
from torchvision import transforms

ROOT_DIRECTORY = Path("/codeexecution")
SUBMISSION_DIRECTORY = ROOT_DIRECTORY / "submission" #output of prediction
DATA_DIRECTORY = ROOT_DIRECTORY / "data" #supplementary
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features" #train chips
MODEL_DIRECTORY = ROOT_DIRECTORY / "assets"
EXT_DIRECTORY = DATA_DIRECTORY / "jrc_extent"

# Pytorch Dataset
class FloodDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self, x_paths, y_paths=None, transforms=None):
        self.data = x_paths
        self.label = y_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Loads a 2-channel image from a chip-level dataframe
        img = self.data.iloc[idx]
        data_dict = {}
        for key, val in img.items():
            if key in ['vv', 'vh', 'elevation', 'extent']:
                try:
                    with rasterio.open(val) as f:
                        array = f.read(1)
                except:
                    array = np.zeros(512, dtype=np.int8)
                if key == "extent":
                    array = np.where(array<=1, array, 0) # filtered out 255
                data_dict[key] = array

        data_stack = []
        for var in ['vv', 'vh', 'elevation', 'extent']:
          try:
            data_stack.append(data_dict[var])
          except KeyError:
            continue
            #print(f"Key {var} is missing!")
        x_arr = np.stack(data_stack, axis=-1) # stack two poplarization together
        
        # Apply data augmentations, if provided
        # if self.transforms:
        #     x_arr = self.transforms(image=x_arr)["image"]
        x_arr = np.transpose(x_arr, [2, 0, 1]) # [N, C, H, W]
        h, w = x_arr.shape[-2], x_arr.shape[-1]

        # Normalization on the tensor:
        mean_para = [-10.98, -17.79, 0.5]
        std_para = [3.44,  3.77, 0.5]
        normalize = transforms.Normalize(mean = mean_para, std = std_para)
        x_arr = torch.from_numpy(x_arr).type(torch.FloatTensor)
        x_arr = normalize(x_arr)

        # Prepare sample dictionary.
        sample = {"chip_id": img.chip_id, "input": x_arr, "metadata": {"size": (h, w)}}
        return sample 




def construct_input_df(chip_ids):
    vv = [INPUT_IMAGES_DIRECTORY / f"{chip_id}_vv.tif" for chip_id in chip_ids]
    vh = [INPUT_IMAGES_DIRECTORY / f"{chip_id}_vh.tif" for chip_id in chip_ids]
    extent = [EXT_DIRECTORY / f"{chip_id}.tif" for chip_id in chip_ids]
    chip_ids = [chip_id for chip_id in chip_ids]
    df = pd.DataFrame({'vv': vv, 'vh': vh, 'extent': extent, 'chip_id': chip_ids})
    return df

def make_predictions(model, input_df):#, chip_id: str):
    """
    Given an image ID, read in the appropriate files and predict a mask of all ones or zeros
    """
    flood_test = FloodDataset(x_paths = input_df[['vh',	'vv', 'extent', 'chip_id']])
    #test_dl = DataLoader(flood_test, batch_size=1, num_workers=1, pin_memory=True)
    predictions = model.predict(flood_test)
    #output_prediction = predictions.astype(np.uint8)
    #output_prediction = [torch.from_numpy(np.array(x).astype(np.int8)) for x in predictions]
    #logger.info("Start looping")
    return predictions
    # for idx, data in tqdm(enumerate(test_dl), miniters=25):
    #     prediction = model.predict(data)
    #     logger.info("success predict")
    #     output_path = SUBMISSION_DIRECTORY / f"{input_df['chip_id'].iloc[idx]}.tif"
    #     logger.info(f"wrtie to {output_path}")
    #     # make our predictions! (you should edit `make_predictions` to do something useful)
    #     #hat = np.array(predict).astype(np.int8)
    #     imwrite(output_path, np.array(prediction).astype(np.int8), dtype=np.uint8)

    # logger.info("check 1")
    # for idx, row in tqdm(input_df.iterrows(), miniters=25, file=sys.stdout, leave=True):
    #     flood_test = FloodDataset(x_paths = row[['vh',	'vv', 'extent', 'chip_id']])
    #     prediction = model.predict(flood_test)
    #     logger.info("success predict")
    #     output_path = SUBMISSION_DIRECTORY / f"{row['chip_id']}.tif"
    #     # make our predictions! (you should edit `make_predictions` to do something useful)
    #     #hat = np.array(predict).astype(np.int8)
    #     imwrite(output_path, np.array(prediction).astype(np.int8), dtype=np.uint8)
    #logger.success(f"... done")
    # for predict in predictions:
    #     np.array(predict).astype(np.int8)

    # output_prediction = [np.array(x).astype(np.int8) for x in predictions]
    # logger.info(f"complete batch predictions")
    # return output_prediction

def dummy_predictions(input_df):
    """
    Given an image ID, read in the appropriate files and predict a mask of all ones or zeros
    """
    flood_test = FloodDataset(x_paths = input_df[['vh',	'vv', 'extent', 'chip_id']])
    #predictions = model.predict(flood_test)
    output_prediction = []
    for i in range(len(input_df)):
        blend = np.array(flood_test[i]['input']).max(axis=0)
        prediction = np.where(blend<1, 1, 0) 
        output_prediction.append(prediction.astype(np.uint8))
    logger.info(f"complete batch predictions")
    return output_prediction


def get_expected_chip_ids():
    """
    Use the input directory to see which images are expected in the submission
    """
    paths = INPUT_IMAGES_DIRECTORY.glob("*.tif")
    # images are named something like abc12.tif, we only want the abc12 part
    ids = list(sorted(set(path.stem.split("_")[0] for path in paths)))
    return ids


def main():
    """
    for each input file, make a corresponding output file using the `make_predictions` function
    """
    chip_ids = get_expected_chip_ids()
    if not chip_ids:
        typer.echo("No input images found!")
        raise typer.Exit(code=1)
    logger.info(f"found {len(chip_ids)} expected image ids; generating predictions for each ...")
    
    input_df = construct_input_df(chip_ids)

    # Load in model:
    model_path = MODEL_DIRECTORY / f"tf_efficientnet_lite4-unetplusplus-imagenet-default-extent.pt"
    model = SemanticSegmentation.load_from_checkpoint(model_path, pretrained=False)
    # need to load model architecture

    # Loop through test dataset
    # create similar DataFrame, then create FloodDataset
    # from model.predict(flood_test)
    # imrite each array to /submission
    predictions = make_predictions(model, input_df) # tensor
    logger.info("success predict")
    #hats = dummy_predictions(input_df) 

    for idx, hat in tqdm(enumerate(predictions), miniters=25, file=sys.stdout, leave=True):
        output_path = SUBMISSION_DIRECTORY / f"{input_df['chip_id'].iloc[idx]}.tif"
        imwrite(output_path, np.array(hat).astype(np.uint8), dtype=np.uint8)
    logger.info("complete!")

if __name__ == "__main__":
    typer.run(main)
