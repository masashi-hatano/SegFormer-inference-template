from argparse import ArgumentParser
from pathlib import Path

from transformers import SegformerForSemanticSegmentation
import torch
import numpy as np
import cv2
import numpy as np
from torch.utils.data import DataLoader
import tqdm

from dataloader.dataset import Dataset
from utils.util import paletteToRGB
from utils.class_names import get_palette


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="/path/to/your_data_dir/")
    parser.add_argument("--save_dir", default="/path/to/output_dir/")
    parser.add_argument("--batch_size", default=1)
    parser.add_argument(
        "--model", default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--size", default=(1024, 2048), help="final output size of segmented img"
    )
    parser.add_argument(
        "--palette",
        default="cityscapes",
        help="Color palette used for segmentation map",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # loarding model and feature extraction
    model = SegformerForSemanticSegmentation.from_pretrained(args.model)

    # Dataset
    dataset = Dataset(args.data_dir)
    # DataLoader
    dataloarder = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    # model to device
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    for cur_iter, (img, file_name) in enumerate(tqdm.tqdm(dataloarder)):
        # img: [B, C, H, W]
        b = img.shape[0]
        with torch.no_grad():
            inputs_tensor = img.cuda()
            outputs = model(inputs_tensor)
            pred = outputs.logits

        # palette to rgb
        pred_rgb = paletteToRGB(pred, args.size, get_palette(args.palette))

        # batch loop
        for batch_idx in range(b):
            # save the segmented and masked images
            path_to_pred_png = Path(args.save_dir, "png", file_name[batch_idx])
            path_to_pred_npy = Path(args.save_dir, "npy", file_name[batch_idx])
            path_to_pred_png.parent.mkdir(parents=True, exist_ok=True)
            path_to_pred_npy.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path_to_pred_png), pred_rgb[batch_idx])
            np.save(str(path_to_pred_npy), pred_rgb[batch_idx])


if __name__ == "__main__":
    main()
