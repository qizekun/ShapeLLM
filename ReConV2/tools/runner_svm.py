import torch
from ReConV2.tools import builder
from ReConV2.utils import misc, dist_utils
from ReConV2.utils.logger import *
import numpy as np
from sklearn.svm import LinearSVC


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC(C=0.075)
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    print_log('Start SVM test... ', logger=logger)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                               builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts)

    if args.use_gpu:
        base_model.to(args.local_rank)
    base_model.eval()

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())

        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(),
                           test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[TEST_SVM] acc = %.4f' % (acc * 100), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
