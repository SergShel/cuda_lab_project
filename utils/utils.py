import numpy as np
import random
import torch
from matplotlib import pyplot as plt
import os
import torchvision.transforms as T
import seaborn as sns
import shutil
from torch.utils.tensorboard import SummaryWriter



def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 42
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def plot_confusion(confusion_matrix, xy_ticklabels=None):
    plt.figure(figsize = (16,8))
    if xy_ticklabels == None:
        conf_plot = sns.heatmap(confusion_matrix, annot=True, fmt='g')
    else:
        conf_plot = sns.heatmap(confusion_matrix, annot=True, fmt='g', xticklabels=xy_ticklabels, yticklabels=xy_ticklabels)
    conf_plot.set(xlabel='True Label', ylabel='Predicted Label')
    plt.show()


def plot_results(ax, train_result, train_eval_result, mode=None):
    """
    Plots a graph side by side, used for train and eval graphs (either accuracy or loss).
    Mode sets the strings for the graph plots
    """
    plt.style.use('seaborn')

    train_result = np.array(train_result)
    ax[0].plot(train_result, c="blue", label=f"Training {mode}", linewidth=3, alpha=0.5)
    ax[0].plot(smooth(train_result, 10), c="red", label=f"Smoothed Training {mode}", linewidth=3, alpha=0.5)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel(f"CE {mode}")
    ax[0].set_title(f"Training {mode}")

    # since our evaluation loss is a nested list
    train_eval_result = np.array(train_eval_result).flatten()
    ax[1].plot(train_eval_result, c="blue", label=f"Evaluation {mode}", linewidth=3, alpha=0.5)
    ax[1].plot(smooth(train_eval_result, 30), c="red", label=f"Smoothed Evaluation {mode}", linewidth=3, alpha=0.5)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel(f"CE {mode}")
    ax[1].set_title(f"Evaluation {mode}")


def get_loader(dataset, dataset_name, train_transform=T.ToTensor(), test_transform=T.ToTensor(), BATCH_SIZE=64):
    """
    Returns the train and test loader for a given dataset.
    Define that dataset name and dataset function (torchvision). Optionally give a transform and batch_size
    """

    root = f"./{dataset_name}/"
    if os.path.exists(root):
        download = False
    else:
        download = True

    train_set = dataset(
        root=root,
        split="train",
        transform=train_transform,
        download=download
    )

    test_set = dataset(
        root=root,
        split="test",
        transform=test_transform,
        download=download
    )

    # enable pin memory for faster transfer between CPU and GPU
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
        )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
        )

    num_labels = torch.unique(train_set.target)
    print(num_labels)

    return train_loader, test_loader, num_labels

def save_model(model, optimizer, epoch, stats, savepath=None):
    """ Saving model checkpoint """

    if(not os.path.exists("models")):
        os.makedirs("models")
    if savepath == None:
        savepath = f"models/checkpoint_{model.__class__.__name__}_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def load_model(model, optimizer, savepath, device):
    """ Loading pretrained checkpoint """

    checkpoint = torch.load(savepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    model = model.to(device)
    optimizer_to(optimizer, device)

    return model, optimizer, epoch, stats

def load_model_by_params(model, optimizer, savepath, device):
    """ Loading pretrained checkpoint """

    checkpoint = torch.load(savepath, map_location=device)
    print(len(checkpoint['model_state_dict']))
    model_list = model.named_parameters()
    for (name, module), c in zip(model_list, checkpoint['model_state_dict']):
        print(name, c)

    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint["epoch"]
    # stats = checkpoint["stats"]
    # model = model.to(device)
    # optimizer_to(optimizer, device)


# Tensorboard config and writter init
def make_tboard_logs(dirname):
    TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", dirname)
    if not os.path.exists(TBOARD_LOGS):
        os.makedirs(TBOARD_LOGS)

    # shutil.rmtree(TBOARD_LOGS)
    writer = SummaryWriter(TBOARD_LOGS)

    print(TBOARD_LOGS)

    return writer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
