import os
import argparse
import glob
import logging

import torch
import numpy as np

from PIL import Image

import modules.midas.utils as utils

import pipeline
from tqdm import tqdm


def load_input_image(input_image_fp):
    return utils.read_image(input_image_fp)


def load_sparse_depth(input_sparse_depth_fp):
    input_sparse_depth = (
        np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
    )
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth


def run(
    depth_predictor,
    nsamples,
    sml_model_path,
    min_pred,
    max_pred,
    min_depth,
    max_depth,
    input_path,
    output_path,
    save_output,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.getLogger(__name__)
    logging.info("Device: %s", device)

    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor,
        nsamples,
        sml_model_path,
        min_pred,
        max_pred,
        min_depth,
        max_depth,
        device,
    )

    # get inputs
    img_names = glob.glob(os.path.join(input_path, "image", "*"))

    # create output folders
    if save_output:
        os.makedirs(os.path.join(output_path, "ga_depth"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "sml_depth"), exist_ok=True)

    for ind, input_image_fp in tqdm(enumerate(img_names)):
        if os.path.isdir(input_image_fp):
            continue

        # print("  processing {} ({}/{})".format(input_image_fp, ind + 1, num_images))

        input_image = load_input_image(input_image_fp)

        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = load_sparse_depth(input_sparse_depth_fp)

        # values in the [min_depth, max_depth] range are considered valid;
        # an additional validity map may be specified
        validity_map = None

        # run method
        output = method.run(input_image, input_sparse_depth, validity_map, device)

        if save_output:
            basename = os.path.splitext(os.path.basename(input_image_fp))[0]

            # saving depth map after global alignment
            utils.write_depth(
                os.path.join(output_path, "ga_depth", basename),
                output["ga_depth"],
                bits=2,
            )

            # saving depth map after local alignment with SML
            utils.write_depth(
                os.path.join(output_path, "sml_depth", basename),
                output["sml_depth"],
                bits=2,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level", type=str, default="info", help="debug, info, warning, error"
    )
    # model parameters
    parser.add_argument(
        "-dp",
        "--depth-predictor",
        type=str,
        default="dpt_hybrid",
        help="Name of depth predictor to use in pipeline.",
    )
    parser.add_argument(
        "-ns",
        "--nsamples",
        type=int,
        default=150,
        help="Number of sparse metric depth samples available.",
    )
    parser.add_argument(
        "-sm",
        "--sml-model-path",
        type=str,
        default="weights/sml_model.dpredictor.dpt_hybrid.nsamples.150.ckpt",
        help="Path to trained SML model weights.",
    )
    # depth parameters
    parser.add_argument(
        "--min-pred",
        type=float,
        default=0.1,
        help="Min bound for predicted depth values.",
    )
    parser.add_argument(
        "--max-pred",
        type=float,
        default=8.0,
        help="Max bound for predicted depth values.",
    )
    parser.add_argument(
        "--min-depth", type=float, default=0.2, help="Min valid depth when evaluating."
    )
    parser.add_argument(
        "--max-depth", type=float, default=5.0, help="Max valid depth when evaluating."
    )

    # I/O paths
    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        default="/home/shuqi/dev/data/void/void_release/void_150/data/classroom6/",
        help="Path to inputs.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="output/classroom6",
        help="Path to outputs.",
    )
    parser.add_argument(
        "--save-output",
        dest="save_output",
        action="store_true",
        help="Save output depth map.",
    )
    parser.set_defaults(save_output=True)

    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.log_level.upper())
    logging.basicConfig(level=numeric_level)
    logging.getLogger(__name__)

    print("Arguments: ")
    for arg in vars(args):
        print("  {} = {}".format(arg, getattr(args, arg) or ""))

    run(
        args.depth_predictor,
        args.nsamples,
        args.sml_model_path,
        args.min_pred,
        args.max_pred,
        args.min_depth,
        args.max_depth,
        args.input_path,
        args.output_path,
        args.save_output,
    )
