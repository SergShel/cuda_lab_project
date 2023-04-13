from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.metrics import confusion_matrix
import torch.utils.tensorboard
from . import utils
import os
import shutil


class Trainer():

    def __init__(self,  model, optimizer, criterion, train_loader,
        valid_loader, train_set, epochs, scheduler=None, sequence=True, tboard_name=None, start_epoch=0,
        all_labels=None, print_intermediate_vals=False, gradient_accumulation=16) -> None:
        super().__init__()

        # needed for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.sequence = sequence
        self.train_set = train_set
        self.gradient_accumulation = gradient_accumulation

        if self.train_set == "coco":
            self.train_fn = self.coco_train
        elif self.train_set == "cityscapes":
            self.train_fn = self.cityscapes_train
        else:
            assert ((train_set == "coco") | (train_set == "cityscapes")),  "Not a valid train set. Valid train sets are coco or cityscapes."

        if scheduler == None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        else:
            self.scheduler = scheduler

        # for saving and loading as well as Tboard directory
        self.save_folder_path = f"models/{self.model.config.__class__.__name__}_{self.model.config.temporal_cell.__name__}/"
        self.model_sizes_string = f"Layers{len(self.model.config.encoder_blocks[0])}_InitDim{self.model.config.encoder_blocks[0][0][1]}"

        # needed for plotting the losses and other metrics
        if tboard_name == None:
            self.tboard = utils.make_tboard_logs(
                f"{self.model.config.__class__.__name__}_{self.model.config.temporal_cell.__name__}/"
                + self.model_sizes_string)
        else:
            self.tboard = utils.make_tboard_logs(tboard_name)

        self.all_labels = all_labels
        self.print_intermediate_vals = print_intermediate_vals
        self.start_epoch = start_epoch



        # losses
        self.train_loss = []
        self.val_loss =  []
        self.loss_iters = []
        self.valid_mIoU = []
        self.valid_mAcc = []
        self.conf_mat = None

    def train_epoch(self, current_epoch):
        """ Training a model for one epoch """

        loss_list = []
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        # grad accumulator running variable
        grad_count=0
        for i, (images, labels) in progress_bar:
            # Clear gradients w.r.t. parameters

            preds, loss, seg_mask = self.train_fn(images, labels)

            # Calculate Loss: softmax --> cross entropy loss
            loss_list.append(loss.item())

            # Getting gradients w.r.t. parameters
            loss = loss / self.gradient_accumulation
            loss.backward()

            grad_count+= images.shape[0]

            # Updating parameters
            if grad_count >= self.gradient_accumulation:
                self.optimizer.step()
                self.optimizer.zero_grad()
                grad_count = 0


            progress_bar.set_description(f"Epoch {current_epoch+1} Iter {i+1}: loss {loss.item():.5f}. ")

            if i == len(self.train_loader)-1:
                mean_loss = np.mean(loss_list)
                progress_bar.set_description(f"Epoch {current_epoch+1} Iter {i+1}: mean loss {mean_loss.item():.5f}. ")

        return mean_loss, loss_list


    @torch.no_grad()
    def eval_model(self):
        """ Evaluating the model for either validation or test """
        correct = 0
        total = 0
        loss_list = []
        Accs = []
        epsilon = 1e-6

        if self.all_labels != None:
            self.conf_mat = torch.zeros(self.all_labels, self.all_labels)
        else:
            self.conf_mat == None

        for images, labels in tqdm(self.valid_loader):

            outputs, loss, seg_mask = self.train_fn(images, labels)

            loss_list.append(loss.item())

            preds = torch.argmax(outputs, dim=2)

            # mIoU
            seg_mask = seg_mask.squeeze(1).view(-1)
            preds = preds.squeeze(1).view(-1)

            if self.all_labels!= None:
                self.conf_mat += confusion_matrix(
                    y_true=seg_mask.cpu().numpy(), y_pred=preds.cpu().numpy(),
                    labels=np.arange(0, self.all_labels, 1)
                )

            # compute mAcc
            num_correct = torch.sum(preds == seg_mask)
            total_predictions = seg_mask.shape[0]
            Accs.append(num_correct/total_predictions)

        iou = self.conf_mat.diag() / (self.conf_mat.sum(axis=1) + self.conf_mat.sum(axis=0) - self.conf_mat.diag() + epsilon)
        mIoU = iou.mean()

        mAcc = sum(Accs) / len(Accs)
        loss = np.mean(loss_list)
        return mIoU, mAcc, loss


    def train_model(self):
        """ Training a model for a given number of epochs"""

        start = time.time()
        self.model = self.model.to(self.device)

        for epoch in range(self.epochs):

            # validation epoch
            self.model.eval()  # important for dropout and batch norms
            mIoU, mAcc, loss = self.eval_model()
            self.valid_mIoU.append(mIoU)
            self.valid_mAcc.append(mAcc)
            self.val_loss.append(loss)

            # if we want to use tensorboard
            if self.tboard !=None:
                self.tboard.add_scalar(f'mIoU/Valid', mIoU, global_step=epoch+self.start_epoch)
                self.tboard.add_scalar(f'mAcc/Valid', mAcc, global_step=epoch+self.start_epoch)
                self.tboard.add_scalar(f'Loss/Valid', loss, global_step=epoch+self.start_epoch)

            # # training epoch
            self.model.train()  # important for dropout and batch norms
            mean_loss, cur_loss_iters = self.train_epoch(epoch)
            self.scheduler.step(self.val_loss[-1])
            self.train_loss.append(mean_loss)

            # if we want to use tensroboard
            if self.tboard != None:
                self.tboard.add_scalar(f'Loss/Train', mean_loss, global_step=epoch+self.start_epoch)

            self.loss_iters = self.loss_iters + cur_loss_iters

            if self.print_intermediate_vals: # and epoch % 5 == 0 or epoch==self.epochs-1):
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"    Train loss: {round(mean_loss, 5)}")
                print(f"    Valid loss: {round(loss, 5)}")
                print(f"    mIoU: {mIoU}%")
                print(f"    mAcc: {mAcc}%")
                print("\n")

            self.save_model(self.start_epoch + epoch)

        end = time.time()
        print(f"Training completed after {(end-start)/60:.2f}min")

    def save_model(self, current_epoch):
        os.makedirs(self.save_folder_path, exist_ok=True)
        # save model
        utils.save_model(
            self.model,
            self.optimizer,
            current_epoch,
            [self.train_loss, self.val_loss, self.loss_iters, self.valid_mIoU, self.valid_mAcc, self.conf_mat],
            savepath=(self.save_folder_path + self.model_sizes_string + f"{self.train_set}_epoch_{current_epoch}.pth"),
            )
        # save model configs as json
        self.model.config.save(path=self.save_folder_path
                               + self.model_sizes_string
                               + f"{self.train_set}.json")


    def load_model(self, load_from):
        assert (load_from == "coco") | (load_from == "cityscapes"), "Invalid argument load from. Either load from citscapes or coco trained model"

        self.model, self.optimizer, self.start_epoch, self.stats = utils.load_model(
            self.model,
            self.optimizer,
            (self.save_folder_path
            + self.model_sizes_string
            + f"{self.train_set}_epoch_{self.start_epoch}.pth"),
            self.device
        )
        self.train_loss, self.val_loss, self.loss_iters, self.valid_mIoU, self.valid_mAcc, self.conf_mat = self.stats

    def count_model_params(self):
        """ Counting the number of learnable parameters in a nn.Module """
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return num_params

    def coco_train(self, images, labels):

        images = images.to(self.device)
        labels = labels.to(self.device).to(torch.long)

        # sequence if necessary for single images
        outputs = self.model(images.unsqueeze(1))

        loss = self.criterion(outputs.squeeze(), labels.squeeze().long())

        # no change to labels but return it since cityscapes has a change to labels
        return outputs, loss, labels

    def cityscapes_train(self, images, labels):

        # label always to device
        images = images.to(self.device)
        labels = (labels[0], labels[1].to(self.device).to(torch.long))

        # extract label tuple
        gt_idx = labels[0]
        assert gt_idx.dtype == torch.int64 , f"Expected to be the index of ground truth, got f{gt_idx} instead."
        seg_mask = labels[1]

        # sequence if necessary for single images
        if self.sequence == True:
            # Forward pass only to get logits/output
            outputs = self.model(images)

            gt_train_data = []
            for i in range(len(gt_idx)):
                # get index for ith batch and append that gt_train tensor
                gt_train_data.append(outputs[i, gt_idx[i]])

            gt_train_data = torch.stack(gt_train_data).unsqueeze(1)
            loss = self.criterion(gt_train_data.squeeze(1), seg_mask.squeeze(1))
            outputs = gt_train_data
        else:

            outputs = self.model(images)
            loss = self.criterion(outputs.squeeze(1), seg_mask.squeeze(1))

        return outputs, loss, seg_mask
