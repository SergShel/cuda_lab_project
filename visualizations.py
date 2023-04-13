"""
Utils methods for data visualization
"""

import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import colors
import shutil
import os
import torchvision
import imageio
import PIL
# from torchvision.utils import draw_segmentation_masks
from src.datasets.cityscapes_loader import cityscapesLoader


COLORS = ["blue", "green", "olive", "red", "yellow", "purple", "orange", "cyan",
          "brown", "pink", "darkorange", "goldenrod", "forestgreen", "springgreen",
          "aqua", "royalblue", "navy", "darkviolet", "plum", "magenta", "slategray",
          "maroon", "gold", "peachpuff", "silver", "aquamarine", "indianred", "greenyellow",
          "darkcyan", "sandybrown"]

VOC_COLORMAP = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
]

# Class for colors
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

colors_arr = [colors.RED, colors.GREEN, colors.YELLOW, colors.BLUE, colors.MAGENTA, colors.CYAN]

# Colored value output if colorized flag is activated.
def getColorEntry(val, args):
    # if not args.colorized:
    #     return ""
    if not isinstance(val, float) or math.isnan(val):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN


def visualize_sequence(sequence, savepath=None, add_title=True, add_axis=False, n_cols=10,
                       size=3, n_channels=3, titles=None, unnorm=False, **kwargs):
    """
    Visualizing a sequence of imgs in a grid like manner.

    Args:
    -----
    sequence: torch Tensor
        Sequence of images to visualize. Shape in (N_imgs, C, H, W)
    savepath: string ir None
        If not None, path where to store the sequence
    add_title: bool
        whether to add a title to each image
    n_cols: int
        Number of images per row in the grid
    size: int
        Size of each image in inches
    n_channels: int
        Number of channels (RGB=3, grayscale=1) in the data
    titles: list
        Titles to add to each image if 'add_title' is True
    """
    # initializing grid
    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols)

    # adding super-title and resizing
    figsize = kwargs.pop("figsize", (3*n_cols, 3*n_rows))
    fig.set_size_inches(*figsize)
    fig.suptitle(kwargs.pop("suptitle", ""))

    if unnorm:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        sequence = sequence * std + mean

    # plotting all frames from the sequence
    ims = []
    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_rows > 1 else ax[col]
        f = sequence[i].permute(1, 2, 0).cpu().detach()
        if(n_channels == 1):
            f = f[..., 0]
        im = a.imshow(f, **kwargs)
        ims.append(im)
        if(add_title):
            if(titles is not None):
                cur_title = "" if i >= len(titles) else titles[i]
                a.set_title(cur_title)
            else:
                a.set_title(f"Image {i}")

    # removing axis
    if(not add_axis):
        for i in range(n_cols * n_rows):
            row, col = i // n_cols, i % n_cols
            a = ax[row, col] if n_rows > 1 else ax[col]
            a.axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    return fig, ax, ims


def add_border(x, color_name, pad=1):
    """
    Adding border to image frames

    Args:
    -----
    x: numpy array
        image to add the border to
    color_name: string
        Name of the color to use
    pad: integer
        number of pixels to pad each side
    """
    b, nc, h, w = x.shape
    zeros = torch.zeros if torch.is_tensor(x) else np.zeros
    px = zeros((b, 3, h+2*pad, w+2*pad))
    color = colors.to_rgb(color_name)
    px[:, 0, :, :] = color[0]
    px[:, 1, :, :] = color[1]
    px[:, 2, :, :] = color[2]
    if nc == 1:
        for c in range(3):
            px[:, c, pad:h+pad, pad:w+pad] = x[:, 0]
    else:
        px[:, :, pad:h+pad, pad:w+pad] = x
    return px


def overlay_segmentations(frames, segmentations, colors, num_classes, alpha=0.7):
    """
    Overlaying the segmentation on a sequence of images
    """
    if frames.max() <= 1:
        frames = frames * 255
    frames = frames.to(torch.uint8)

    imgs = []
    for frame, segmentation in zip(frames, segmentations):
        img = overlay_segmentation(frame, segmentation, colors, num_classes, alpha)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


# def overlay_segmentation(img, segmentation, colors, num_classes, alpha=0.7):
#     """
#     Overlaying the segmentation on an image
#     """
#     if img.max() <= 1:
#         img = img * 255
#     img = img.to(torch.uint8)
#     seg_masks = (segmentation[0] == torch.arange(num_classes)[:, None, None].to(segmentation.device))
#     img_with_seg = draw_segmentation_masks(
#             img,
#             masks=seg_masks,
#             alpha=alpha,
#             colors=colors
#         )
#     return img_with_seg / 255


def overlay_instances(frames, instances, colors, alpha):
    """
    Overlay instance segmentations on a sequence of images
    """
    if colors[0] != "white":  # background should always be white
        colors = ["white"] + colors
    if frames.max() <= 1:
        frames = frames * 255
    frames = frames.to(torch.uint8)

    imgs = []
    for frame, instance in zip(frames, instances):
        img = overlay_instance(frame, instance, colors, alpha)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


def overlay_instance(img, instance, colors, alpha=0.7):
    """
    Overlaying the segmentation on an image
    """
    if colors[0] != "white":  # background should always be white
        colors = ["white"] + colors
    if img.max() <= 1:
        img = img * 255
    img = img.to(torch.uint8)
    instance_ids = instance.unique()
    instance_masks = (instance[0] == instance_ids[:, None, None].to(instance.device))
    cur_colors = [colors[idx.item()] for idx in instance_ids]
    # img_with_seg = draw_segmentation_masks(
    #         img,
    #         masks=instance_masks,
    #         alpha=alpha,
    #         colors=cur_colors
    #     )
    # return img_with_seg / 255


def qualitative_evaluation(imgs, targets, preds, unnorm=True):
    """
    Displaying the original images, target segmentation, and predicted segmentation
    """
    dataset_root_dir = "/home/nfs/inf6/data/datasets/cityscapes/"
    val_ds = cityscapesLoader(root=dataset_root_dir, split='val', is_transform=True, silent=True)

    # targets_vis = (targets * 255).long()
    # targets_vis = overlay_segmentations(
    #         frames=imgs,
    #         segmentations=targets_vis,
    #         colors=VOC_COLORMAP,
    #         num_classes=20,
    #         alpha=1
    #     )
    # preds_vis = overlay_segmentations(
    #         frames=imgs,
    #         segmentations=preds,
    #         colors=VOC_COLORMAP,
    #         num_classes=20,
    #         alpha=1
    #     )

    if unnorm:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        imgs = imgs * std + mean

    for i in range(3):
        decoded_pred = val_ds.decode_segmap(preds[i].cpu().numpy())
        decoded_label = val_ds.decode_segmap(targets[i][0].cpu().numpy())
        decoded_img = imgs[i].cpu().numpy()

        


            

    imgs, targets, preds = imgs[:6], targets[:6], preds[:6]
    fig, ax = plt.subplots(nrows=3, ncols=6)
    fig.set_size_inches(30, 10)
    ax[0, 0].set_ylabel("Images", fontsize=24)
    ax[1, 0].set_ylabel("Targets", fontsize=24)
    ax[2, 0].set_ylabel("Predictions", fontsize=24)
    for i in range(3):
        decoded_pred = val_ds.decode_segmap(preds[i].cpu().numpy())
        decoded_label = val_ds.decode_segmap(targets[i][0].cpu().numpy())
        decoded_img = imgs[i].cpu().permute(1, 2, 0).numpy()
        ax[0, i].imshow(decoded_img)
        ax[1, i].imshow(decoded_label)
        ax[2, i].imshow(decoded_pred)
    for aa in ax:
        for a in aa:
            a.set_yticks([], [])
            a.set_xticks([], [])
    plt.tight_layout()
    return fig, ax

class CityscapesVisualizer(object):
    

    def __init__(self, n_classes=19):
        self.n_classes = n_classes
        self.label_colours = self.get_label_colours()

    def get_label_colours(self):
        colors = [  # [  0,   0,   0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ]

        return dict(zip(range(19), colors))


    def decode_segmap(self, temp):

        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def prepare_img(self, img, unnorm=True):
        if unnorm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            img = img * std + mean
        img = img.cpu().permute(1, 2, 0).numpy()
        return img
        
@torch.no_grad()
def vis_seq(model, loader):
    visualizer = vis.CityscapesVisualizer()
    model.eval()
    for i, (imgs, targets) in enumerate(loader):
        imgs = imgs.cuda()
        preds = model(imgs)
        print(preds.shape)
        for i in range(preds.shape[0]):
            decoded_seq = get_decoded_img_seq(preds[i])

            for j in range(len(decoded_seq)):
                plt.imshow(decoded_seq[j])
                plt.show()

            break
        break
    return 

@torch.no_grad()
def save_vis_seq(model, loader, model_name="default"):
    if not os.path.exists("imgs"):
        os.makedirs("imgs")
    if not os.path.exists(f"imgs/{model_name}"):
        os.makedirs(f"imgs/{model_name}")

    visualizer = CityscapesVisualizer()
    model.eval()
    for k, (imgs, targets) in enumerate(loader):
        if not os.path.exists(f"imgs/{model_name}/{k}"):
            os.makedirs(f"imgs/{model_name}/{k}")
        if not os.path.exists(f"imgs/{model_name}/{k}/original"):
            os.makedirs(f"imgs/{model_name}/{k}/original")
        if not os.path.exists(f"imgs/{model_name}/{k}/predicted"):
            os.makedirs(f"imgs/{model_name}/{k}/predicted")
        imgs = imgs.cuda()
        preds = model(imgs)
        print(f"{k}: " + str(preds.shape))
        for i in range(preds.shape[0]):

            decoded_seq = get_decoded_img_seq(preds[i])

            for j in range(len(decoded_seq)):
                torchvision.utils.save_image(torch.from_numpy(decoded_seq[j].transpose(2,0,1)), os.path.join(os.getcwd(), "imgs", f"{model_name}", f"{k}", "predicted", f"imgs_{j}.png"))

                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                unnorm_imgs = imgs.cpu() * std + mean

                torchvision.utils.save_image(unnorm_imgs[i, j], os.path.join(os.getcwd(), "imgs", f"{model_name}", f"{k}", "original", f"imgs_{j}.png"))

        if k > 20:
            break

    return 

def get_decoded_img_seq(preds):
    result = []
    visualizer = CityscapesVisualizer()
    predicted_class = torch.argmax(preds, dim=1)
    for j in range(preds.shape[0]):
        decoded_pred = visualizer.decode_segmap(predicted_class[j].cpu().numpy())
        result.append(decoded_pred)
        #torchvision.utils.save_image(torch.from_numpy(decoded_pred.transpose(2,0,1)), os.path.join(os.getcwd(), "imgs", "training", f"imgs_{j}.png"))
    return result




def create_gifs(model_name="default", mode="side-by-side", transparency=0.5, fps=8):

    allowed_modes = ["side-by-side", "overlay"]
    if mode not in allowed_modes:
        raise ValueError(f"mode must be one of {alloud_modes}")
    imgs_root=f"imgs/{model_name}"
    dirlist = [ item for item in os.listdir(imgs_root) if os.path.isdir(os.path.join(imgs_root, item)) ]
    # remove "gifs"-folder from dirlist
    dirlist = [item for item in dirlist if item != "gifs"]

    if not os.path.exists(f"imgs/{model_name}/gifs"):
        os.makedirs(f"imgs/{model_name}/gifs")



    if mode == "side-by-side":
        for i in range(len(dirlist)):
            images = []
            for j in range(12):
                original = PIL.Image.open(f"imgs/{model_name}/{dirlist[i]}/original/imgs_{j}.png")
                prediction = PIL.Image.open(f"imgs/{model_name}/{i}/predicted/imgs_{j}.png")

                (width1, height1) = original.size
                (width2, height2) = prediction.size

                result_width = width1 + width2
                result_height = max(height1, height2)

                result = PIL.Image.new('RGB', (result_width, result_height))
                result.paste(im=original, box=(0, 0))
                result.paste(im=prediction, box=(width1, 0))
                images.append(result)
            imageio.mimsave(f"imgs/{model_name}/gifs/{i}.gif", images, fps=fps)

    elif mode == "overlay":
        for i in dirlist:
            images = []
            for j in range(12):
                background = PIL.Image.open(f"imgs/{model_name}/{i}/original/imgs_{j}.png")
                foreground  = PIL.Image.open(f"imgs/{model_name}/{i}/predicted/imgs_{j}.png")
                foreground.putalpha(int(255*(1-transparency))) 
                background.paste(foreground, (0, 0), mask=foreground)
                images.append(background)
            imageio.mimsave(f"imgs/{model_name}/gifs/{i}.gif", images, fps=fps)
