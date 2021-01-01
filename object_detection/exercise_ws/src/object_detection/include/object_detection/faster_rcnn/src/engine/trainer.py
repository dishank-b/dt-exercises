import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.utils import tensorboard
from torchvision import transforms as T
# from torchviz import make_dot

from ..architecture import build_model
from ..datasets import build_dataset
from ..eval.evaluator import Evaluator
from ..tracker import MultiObjTracker
from ..utils import Instances, Boxes, utils


class General_Solver(object):
    """
    This is a General Solver class which comprises of
    all different parts realted to training or
    inference.
    """
    def __init__(self, cfg, mode, args):
        super(General_Solver, self).__init__()

        self.cfg = cfg

        self.device = torch.device("cuda") if (
            torch.cuda.is_available()
            and cfg.USE_CUDA) else torch.device("cpu")
        print("--- Using the device for training: {} \n".format(self.device))

        self.setup_dirs(cfg, args.name)

        if mode == "train":
            file = open(os.path.join(self.exp_dir, "config.yaml"), 'w')
            cfg.dump(stream=file, default_flow_style=False)

        self.model = self.get_model(cfg)

        self.optimizer, self.lr_scheduler = self.build_optimizer(cfg)

        #From where to start the training
        self.epoch = 0

        #Very important parameter for internal model calculations.
        self.is_training = True

        if mode == "test" or args.weights or args.resume:
            self.load_checkpoint(cfg, args)

        self.train_loader, self.val_loader = self.get_dataloader(cfg, mode)

        self.tb_writer = tensorboard.SummaryWriter(
            os.path.join(self.exp_dir, "tf_summary"))

    def get_model(self, cfg):
        print("--- Building the Model \n")
        # print(cfg.ROI_HEADS.SCORE_THRESH_TEST)
        model = build_model(cfg.ARCHITECTURE.MODEL)(cfg)
        model = model.to(self.device)

        return model

    def save_checkpoint(self):
        model_path = os.path.join(
            self.exp_dir, "models",
            "epoch_" + str(self.epoch).zfill(5) + ".model")
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'cfg': self.cfg
            }, model_path)

    def load_checkpoint(self, cfg, args):
        if args.weights:
            print("--- Using pretrainted weights from: {}".format(
                args.weights))
            checkpoint = torch.load(args.weights)
            self.model.load_state_dict(checkpoint['model_state_dict'],
                                       strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # print("Optimizer State: ", self.optimizer.state_dict())

        elif args.resume:
            print("    :Resuming the training \n")
            self.epoch = args.epoch  #With which epoch you want to resume the training.
            checkpoint = torch.load(model_save_dir + "/epoch_" +
                                    str(self.epoch).zfill(5) + '.model')

            self.model.load_state_dict(checkpoint['model_state_dict'],
                                       strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        else:
            print("--- Loading weights for testing")
            weight_path = os.path.join(
                self.exp_dir, "models",
                "epoch_" + str(args.epoch).zfill(5) + '.model')
            checkpoint = torch.load(weight_path)
            self.model.load_state_dict(checkpoint['model_state_dict'],
                                       strict=False)

    def build_optimizer(self, cfg):
        if cfg.SOLVER.OPTIM.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=cfg.TRAIN.LR,
                                   weight_decay=1e-4)
        elif cfg.SOLVER.OPTIM.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=cfg.TRAIN.LR,
                                  momentum=cfg.TRAIN.MOMENTUM,
                                  weight_decay=1e-4)
        else:
            raise ValueError('Optimizer must be one of \"sgd\" or \"adam\"')

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN.MILESTONES,
            gamma=cfg.TRAIN.LR_DECAY,
            last_epoch=-1)

        return optimizer, lr_scheduler

    def get_dataloader(self, cfg, mode):
        dataset = build_dataset(cfg.DATASET.NAME)
        dataset_path = cfg.DATASET.PATH
        assert os.path.exists(dataset_path), "Dataset path doesn't exist."

        transform = utils.image_transform(
            cfg)  # this is tranform to normalise/standardise the images
        batch_size = cfg.TRAIN.BATCH_SIZE

        print("--- Loading Training Dataset \n ")

        dataset = dataset(dataset_path, transform=transform,
                          cfg=cfg)  #---- Dataloader

        train_len = int(0.9 * len(dataset))
        val_len = len(dataset) - train_len

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_len, val_len])

        print("--- Data Loaded---")
        print("Number of Images in Train Dataset: {} \n".format(
            len(train_dataset)))
        print("Number of Images in Val Dataset: {} \n".format(
            len(val_dataset)))
        print("Number of Classes in Dataset: {} \n".format(
            cfg.INPUT.NUM_CLASSES))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=cfg.TRAIN.DSET_SHUFFLE,
            collate_fn=dataset.collate_fn,
            drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 collate_fn=dataset.collate_fn,
                                                 drop_last=True)

        return train_loader, val_loader

    def setup_dirs(self, cfg, name):
        self.exp_dir = os.path.join(cfg.LOGS.BASE_PATH, name)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
            os.mkdir(os.path.join(self.exp_dir, "models"))
            os.mkdir(os.path.join(self.exp_dir, "results"))
            os.mkdir(os.path.join(self.exp_dir, "tf_summary"))
        else:
            "Directory exists, You may want to resume instead of starting fresh."

    def train_step(self, batch_sample):
        in_images = batch_sample['image'].to(self.device)
        target = [x.to(self.device) for x in batch_sample['target']]
        img_paths = batch_sample['image_path']

        rpn_proposals, instances, rpn_losses, detection_losses = self.model(
            in_images, target, self.is_training)
        # print("Images:", img_paths)
        # print("Target:", [len(x) for x in target])
        # print("Output:", [len(x) for x in instances])

        loss_dict = {}
        loss_dict.update(rpn_losses)
        loss_dict.update(detection_losses)

        loss = 0.0
        for k, v in loss_dict.items():
            loss += v
        loss_dict.update({'tot_loss': loss})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_dict

    def validation_step(self):
        self.model.eval()
        val_loss = {}
        with torch.no_grad():
            for idx, batch_sample in enumerate(self.val_loader):
                in_images = batch_sample['image'].to(self.device)
                target = [x.to(self.device) for x in batch_sample['target']]

                rpn_proposals, instances, rpn_losses, detector_losses = self.model(
                    in_images, target, self.is_training)

                loss_dict = {}
                loss_dict.update(rpn_losses)
                loss_dict.update(detector_losses)

                loss = 0.0
                for k, v in loss_dict.items():
                    loss += v
                loss_dict.update({'tot_loss': loss})

                for key, value in loss_dict.items():
                    if len(val_loss) < len(loss_dict):
                        val_loss[key] = [loss_dict[key].item()]
                    val_loss[key].append(loss_dict[key].item())

                # utils.tb_logger(in_images, tb_writer, rpn_proposals, instances, "Validation")

            for key, value in val_loss.items():
                val_loss[key] = np.mean(val_loss[key])

        self.model.train()
        return val_loss

    def train(self, epochs, saving_freq):
        assert self.model.training
        self.is_training = True

        for name, module in self.model.named_children():
            for p_name, p in module.named_parameters():
                print(p_name, p.requires_grad)

        while self.epoch <= epochs:
            running_loss = {}
            print("Epoch | Iteration | Loss ")

            for idx, batch_sample in enumerate(self.train_loader):
                # print("Iteration:", idx)
                self.model.backbone.freeze_bn()
                loss_dict = self.train_step(batch_sample)

                if idx % 10 == 0:
                    # with torch.no_grad():
                    #----------- Logging and Printing ----------#
                    print("{:<8d} {:<9d} {:<7.4f}".format(
                        self.epoch, idx, loss_dict['tot_loss'].item()))

                for loss_name, value in loss_dict.items():
                    self.tb_writer.add_scalar('Loss/' + loss_name,
                                              value.item())

                for key, value in loss_dict.items():
                    if idx == 0:
                        running_loss[key] = 0.0
                    running_loss[key] = 0.9 * running_loss[
                        key] + 0.1 * loss_dict[key].item()
                # utils.tb_logger(in_images, tb_writer, rpn_proposals, instances, "Training")
                #------------------------------------------------#

            val_loss = self.validation_step()

            for key in val_loss.keys():
                self.tb_writer.add_scalars('loss/' + key, {
                    'validation': val_loss[key],
                    'train': running_loss[key]
                }, self.epoch)

            # print("Epoch ---- {} ".format(self.epoch))
            print("Epoch      Training   Validation")
            print("{} {:>13.4f}    {:0.4f}".format(self.epoch,
                                                   running_loss['tot_loss'],
                                                   val_loss['tot_loss']))

            ## Decaying learning rate
            self.lr_scheduler.step()

            print("Epoch Complete: ", self.epoch)
            # # Saving at the end of the epoch
            if self.epoch % saving_freq == 0:
                self.save_checkpoint()

            self.epoch += 1

        self.tb_writer.close()

    def test(self):
        self.model.eval()
        self.is_training = False
        # evaluator = Evaluator(4)
        evaluator = Evaluator(2)
        print("--- Running Inference")

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = len(self.val_loader)
        timings=np.zeros((repetitions,1))
        
        with torch.no_grad():

            count = 0
            for idx, batch_sample in enumerate(self.train_loader):
                in_images = batch_sample['image'].to(self.device)
                targets = [x.to(self.device) for x in batch_sample['target']]
                img_paths = batch_sample['image_path']
                _ = self.model(
                    in_images, targets, self.is_training)
                count +=1
                if count==10:
                    break

            for idx, batch_sample in enumerate(self.val_loader):
                in_images = batch_sample['image'].to(self.device)
                targets = [x.to(self.device) for x in batch_sample['target']]
                img_paths = batch_sample['image_path']

                start = time.time()
                starter.record()
                rpn_proposals, instances, proposal_losses, detector_losses = self.model(
                    in_images, targets, self.is_training)
                print("CPU TIME: " ,time.time() - start)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[idx] = curr_time

                evaluator.evaluate(in_images, instances, targets)
                targets = [x.numpy() for x in targets]
                instances = [x.numpy() for x in instances]
                rpn_proposals = [x.numpy() for x in rpn_proposals]

                utils.disk_logger(in_images, os.path.join(self.exp_dir,"results"), instances, rpn_proposals, img_paths)
        
        print(np.sum(timings)/repetitions)

        evaluator.print()
        # evaluator.plot(self.exp_dir)

