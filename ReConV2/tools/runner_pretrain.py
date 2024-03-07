import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC

from ReConV2.tools import builder
from ReConV2.datasets import data
from ReConV2.utils.config import *
from ReConV2.utils.logger import *
from ReConV2.utils import dist_utils
from ReConV2.utils.AverageMeter import AverageMeter
from ReConV2.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC(C=0.075)
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def check_nan_gradients(grad):
    if torch.isnan(grad).any():
        return torch.zeros_like(grad)
    return grad


def run_net(args, config):
    logger = get_logger(args.log_name)

    # build dataset
    config.dataset.train.others.with_color = config.model.with_color
    config.dataset.train.others.img_queries = config.model.img_queries
    config.dataset.train.others.text_queries = config.model.text_queries
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)

    # build model
    device = torch.device("cuda", args.local_rank)

    base_model = builder.model_builder(config.model)
    base_model.to(device)

    for p in base_model.named_parameters():
        if p[1].requires_grad is True:
            p[1].register_hook(check_nan_gradients)
            print(p[0])

    if args.local_rank == 0:
        total_params = sum(p.numel() for p in base_model.parameters()) / 1e6
        print_log(f"Total number of parameters: {total_params:.2f}M", logger=logger)
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad) / 1e6
        print_log(f"Total number of trainable parameters: {trainable_params:.2f}M", logger=logger)

    if args.local_rank == 0:
        modelnet40_loader = data.make_modelnet40test(config)
        objaverse_lvis_loader = data.make_objaverse_lvis(config)
        scanobjectnn_loader = data.make_scanobjectnntest(config)
    else:
        modelnet40_loader = None
        objaverse_lvis_loader = None
        scanobjectnn_loader = None

    # trainval
    # training
    base_model.zero_grad()
    trainer = Trainer(args.local_rank, args, config, base_model, train_sampler, train_dataloader,
                      device, logger, modelnet40_loader=modelnet40_loader, scanobjectnn_loader=scanobjectnn_loader,
                      objaverse_lvis_loader=objaverse_lvis_loader)
    if args.contrast:
        trainer.load_from_reconstruct(args.ckpts)
    if args.resume:
        trainer.load_from_checkpoint(os.path.join(args.experiment_path, 'ckpt-last.pth'))
    trainer.model_parallel()
    trainer.build_opti_sche()
    if args.resume:
        builder.resume_optimizer(trainer.optimizer, args, logger=logger)

    test = False if args.reconstruct else True
    trainer.train(test=test)


class Trainer(object):
    def __init__(self, rank, args, config, model, train_sampler, train_loader, device, logger,
                 modelnet40_loader=None, scanobjectnn_loader=None, objaverse_lvis_loader=None):
        self.rank = rank
        self.args = args
        self.config = config
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.train_sampler = train_sampler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.epoch = 0
        self.device = device
        self.logger = logger
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_scanobjectnn_acc = 0
        self.best_lvis_acc = 0
        if args.contrast:
            self.training_type = 'contrast'
        elif args.reconstruct:
            self.training_type = 'reconstruct'
        else:
            self.training_type = 'all'

    def model_parallel(self):
        if self.args.distributed:
            # Sync BN
            if self.args.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                print_log('Using Synchronized BatchNorm ...', logger=self.logger)
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[self.args.local_rank],
                                                             find_unused_parameters=True,
                                                             output_device=self.args.local_rank)
            print_log('Using Distributed Data parallel ...', logger=self.logger)
        else:
            print_log('Using Data parallel ...', logger=self.logger)
            self.model = nn.DataParallel(self.model)

    def build_opti_sche(self):
        self.optimizer, self.scheduler = builder.build_opti_sche(self.model, self.config)

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        base_model = checkpoint['base_model']
        state_dict = {k.replace('module.', ''): v for k, v in base_model.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_modelnet40_overall_acc = checkpoint.get('best_modelnet40_overall_acc', 0.)
        self.best_modelnet40_class_acc = checkpoint.get('best_modelnet40_class_acc', 0.)
        self.best_scanobjectnn_acc = checkpoint.get('best_scanobjectnn_acc', 0.)
        self.best_lvis_acc = checkpoint.get('best_lvis_acc', 0.)

        print_log("Loaded checkpoint from {}".format(path), logger=self.logger)
        print_log("----Epoch: {0}".format(self.epoch), logger=self.logger)
        print_log("----Best modelnet40 overall acc: {}".format(self.best_modelnet40_overall_acc), logger=self.logger)
        print_log("----Best modelnet40 class acc: {}".format(self.best_modelnet40_class_acc), logger=self.logger)
        print_log("----Best scanobjectnn acc: {}".format(self.best_scanobjectnn_acc), logger=self.logger)
        print_log("----Best lvis acc: {}".format(self.best_lvis_acc), logger=self.logger)

    def load_from_reconstruct(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        base_model = checkpoint['base_model']
        state_dict = {k.replace('module.', ''): v for k, v in base_model.items()}
        model_dict = self.model.state_dict()
        for k, v in state_dict.items():
            if 'global_blocks' in k:
                state_dict[k] = model_dict[k.replace('module.', '')]
        incompatible = self.model.load_state_dict(state_dict, strict=False)

        if self.rank == 0:
            if incompatible.missing_keys:
                print_log('missing_keys', logger='PointTransformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='PointTransformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='PointTransformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='PointTransformer'
                )

        print_log("Successfully load checkpoint from {}".format(path), logger=self.logger)

    def train_one_epoch(self, epoch):
        self.model.train()

        if self.args.distributed:
            self.train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        n_batches = len(self.train_loader)
        for idx, (pc, img, text, uid) in enumerate(self.train_loader):

            data_time.update(time.time() - batch_start_time)
            npoints = self.config.dataset.train.others.npoints
            points = pc.to(self.device)

            assert points.size(1) == npoints
            img = img.to(self.device)
            text = text.to(self.device)

            loss = self.model(points, img, text, self.training_type)

            try:
                loss.backward()
            except:
                loss = loss.mean()
                loss.backward()

            # forward
            self.optimizer.step()
            self.model.zero_grad()

            if self.args.distributed:
                loss = dist_utils.reduce_tensor(loss, self.args)
                torch.cuda.synchronize()

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' %
                      (epoch, self.config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                       loss, self.optimizer.param_groups[0]['lr']), logger=self.logger)
        if isinstance(self.scheduler, list):
            for item in self.scheduler:
                item.step(epoch)
        else:
            self.scheduler.step(epoch)
        epoch_end_time = time.time()

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, loss,
                   self.optimizer.param_groups[0]['lr']), logger=self.logger)

    def save_model(self, name):
        torch.save({
            "base_model": self.model.state_dict(),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "epoch": self.epoch,
            "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
            "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
            "best_scanobjectnn_acc": self.best_scanobjectnn_acc,
            "best_lvis_acc": self.best_lvis_acc,
        }, os.path.join(self.args.experiment_path, '{}.pth'.format(name)))
        print_log(f"Save checkpoint at {os.path.join(self.args.experiment_path, '{}.pth'.format(name))}",
                  logger=self.logger)

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct

    def train(self, test=True):
        for epoch in range(self.epoch, self.config.max_epoch + 1):
            self.epoch = epoch
            self.train_one_epoch(epoch)
            if self.rank == 0:
                self.save_model('ckpt-last')
                if test:
                    self.test_modelnet40()
                    self.test_scanobjectnn()
                    self.test_objaverse_lvis()
                if epoch % 25 == 0 and epoch >= 250:
                    self.save_model('epoch_{}'.format(self.epoch))

    def train_contrast(self):

        for epoch in range(self.epoch, self.config.max_epoch + 1):
            self.epoch = epoch
            self.train_one_epoch(epoch)
            if self.rank == 0:
                self.save_model('ckpt-last')
                self.test_modelnet40()
                self.test_scanobjectnn()
                self.test_objaverse_lvis()
                if epoch % 25 == 0 and epoch >= 250:
                    self.save_model('epoch_{}_contrast'.format(self.epoch))

    def test_modelnet40(self):
        self.model.eval()
        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).to(self.device)
        per_cat_correct = torch.zeros(40).to(self.device)
        per_cat_count = torch.zeros(40).to(self.device)
        ratio = self.config.modelnet40.ratio

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.modelnet40_loader:
                pcs = data['pcs'].to(self.device)
                _, _, img_token, text_token = self.model.module.inference(pcs)
                img_pred_feat = torch.mean(img_token, dim=1)
                text_pred_feat = torch.mean(text_token, dim=1)
                pred_feat = img_pred_feat + text_pred_feat * ratio
                logits = pred_feat @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

                for i in range(40):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()
        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1, 3, 5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum() * 100
        per_cat_acc = per_cat_correct / per_cat_count * 100

        if overall_acc > self.best_modelnet40_overall_acc:
            self.best_modelnet40_overall_acc = overall_acc
            self.save_model('best_modelnet40_overall')
        if per_cat_acc.mean() > self.best_modelnet40_class_acc:
            self.best_modelnet40_class_acc = per_cat_acc.mean()
            self.save_model('best_modelnet40_class')

        print_log('Test ModelNet40: overall acc: {:.2f}({:.2f}) class_acc: {:.2f}({:.2f})'.format(overall_acc,
                                                                                                  self.best_modelnet40_overall_acc,
                                                                                                  per_cat_acc.mean(),
                                                                                                  self.best_modelnet40_class_acc),
                  logger=self.logger)
        print_log(
            'Test ModelNet40: top1_acc: {:.2f} top3_acc: {:.2f} top5_acc: {:.2f}'.format(topk_acc[0].item(),
                                                                                         topk_acc[1].item(),
                                                                                         topk_acc[2].item()),
            logger=self.logger)

    def test_objaverse_lvis(self):
        self.model.eval()
        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).to(self.device)
        per_cat_correct = torch.zeros(1156).to(self.device)
        per_cat_count = torch.zeros(1156).to(self.device)
        ratio = self.config.objaverse_lvis.ratio

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in tqdm(self.objaverse_lvis_loader):
                pcs = data['pcs'].to(self.device)
                _, _, img_token, text_token = self.model.module.inference(pcs)
                img_pred_feat = torch.mean(img_token, dim=1)
                text_pred_feat = torch.mean(text_token, dim=1)
                pred_feat = img_pred_feat + text_pred_feat * ratio
                logits = pred_feat @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)
                # calculate per class accuracy
                for i in torch.unique(labels):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()
        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1, 3, 5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum() * 100
        per_cat_acc = per_cat_correct / per_cat_count * 100

        if overall_acc > self.best_lvis_acc:
            self.best_lvis_acc = overall_acc
            self.save_model('best_lvis')
        print_log(
            'Test ObjaverseLVIS: overall acc: {:.2f}({:.2f}) class_acc: {:.2f}'.format(overall_acc,
                                                                                       self.best_lvis_acc,
                                                                                       per_cat_acc.mean()),
            logger=self.logger)
        print_log(
            'Test ObjaverseLVIS: top1_acc: {:.2f} top3_acc: {:.2f} top5_acc: {:.2f}'.format(topk_acc[0].item(),
                                                                                            topk_acc[1].item(),
                                                                                            topk_acc[2].item()),
            logger=self.logger)

    def test_scanobjectnn(self):
        self.model.eval()
        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.device)
        per_cat_correct = torch.zeros(15).to(self.device)
        per_cat_count = torch.zeros(15).to(self.device)
        ratio = self.config.scanobjectnn.ratio

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.scanobjectnn_loader:
                pcs = data['pcs'].to(self.device)
                _, _, img_token, text_token = self.model.module.inference(pcs)
                img_pred_feat = torch.mean(img_token, dim=1)
                text_pred_feat = torch.mean(text_token, dim=1)
                pred_feat = img_pred_feat + text_pred_feat * ratio
                logits = pred_feat @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)
                # calculate per class accuracy
                for i in range(15):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()

        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1, 3, 5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum() * 100
        per_cat_acc = per_cat_correct / per_cat_count * 100

        if overall_acc > self.best_scanobjectnn_acc:
            self.best_scanobjectnn_acc = overall_acc
            self.save_model('best_scanobjectnn')
        print_log(
            'Test ScanObjectNN: overall acc: {:.2f}({:.2f}) class_acc: {:.2f}'.format(overall_acc,
                                                                                      self.best_scanobjectnn_acc,
                                                                                      per_cat_acc.mean()),
            logger=self.logger)
        print_log(
            'Test ScanObjectNN: top1_acc: {:.2f} top3_acc: {:.2f} top5_acc: {:.2f}'.format(topk_acc[0].item(),
                                                                                           topk_acc[1].item(),
                                                                                           topk_acc[2].item()),
            logger=self.logger)
