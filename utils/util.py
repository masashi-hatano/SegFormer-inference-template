import numpy as np
from torch.nn import functional as F

def convertToRGB(index_colored_numpy, palette, n_colors=19):
    if n_colors is None:
        n_colors = palette.shape[0]
    reduced = index_colored_numpy.copy()
    reduced[index_colored_numpy > n_colors] = 0
    expanded_img = np.eye(n_colors, dtype=np.int32)[reduced] # [B, H, W, n_colors]
    use_palette = palette[:n_colors] # [n_colors, 1]
    return np.dot(expanded_img, use_palette).astype(np.uint8)

def paletteToRGB(pred, size, palette):
    pred = F.interpolate(input=pred, size=size, mode='bilinear', align_corners=False)
    pred = np.asarray(np.argmax(pred.cpu(), axis=1), dtype=np.uint8)
    pred_rgb = convertToRGB(pred, palette)
    return pred_rgb