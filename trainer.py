import torch
import torch.nn as nn
from tiny_yolo import tiny_yolo
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import torch.optim as optim
from util import *
from dataset import *
import tqdm


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.config = config
        self.epochs = self.config.epochs

        cuda = torch.cuda.is_available() and self.config.use_gpu is True
        self.model = tiny_yolo(self.config)
        self.model.load_weights(self.config.weightfile)
       #self.model = SE_yolo(self.config, pre_model)

        print(self.model)
        print('[*] Number of model parameters: {:,}'.format(sum([p.data.nelement() for p in self.model.parameters()])))

        if cuda:
            self.model = self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.init_lr/self.config.batch_size, momentum=self.config.momentum, dampening=0, weight_decay=self.config.decay*self.config.batch_size)
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def train(self):

        # loading training data
        t0 =time.time()
        train_path = self.config.train_txt
        dataloader = torch.utils.data.DataLoader(
            ListDataset(train_path), batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.n_cpu
        )

        best_model_wts = self.model.state_dict()
        self.model.train()

        for epoch in range(1,self.epochs+1):
            for batch_i, (_, imgs, targets) in enumerate(dataloader):
                t2= time.time()
                imgs = Variable(imgs.type(self.Tensor))
                targets = Variable(targets.type(self.Tensor), requires_grad=False)

                self.optimizer.zero_grad()

                loss = self.model(imgs, targets)

                loss.backward()
                self.optimizer.step()

                print(
                    "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                    % (
                        epoch,
                        self.config.epochs,
                        batch_i,
                        len(dataloader),
                        self.model.losses["x"],
                        self.model.losses["y"],
                        self.model.losses["w"],
                        self.model.losses["h"],
                        self.model.losses["conf"],
                        self.model.losses["cls"],
                        loss.item(),
                        self.model.losses["recall"],
                        self.model.losses["precision"],
                    )
                )

                self.model.seen += imgs.size(0)

            if epoch % self.config.checkpoint_interval == 0:
                self.model.save_weights("%s/%d.weights" % (self.config.checkpoint_dir, epoch))
                #torch.save(self.model, os.path.join(self.config.checkpoint_dir, "epoch_" + str(epoch) + ".pth.tar"))

        self.model.load_state_dict(best_model_wts)


    def test(self):
        test_path = self.config.test_txt
        dataset = ListDataset(test_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False,
                                                 num_workers=self.config.n_cpu)

        self.model.eval()
        num_classes = self.config.class_num

        all_detections = []
        all_annotations = []

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

            imgs = Variable(imgs.type(self.Tensor))

            with torch.no_grad():
                outputs = self.model(imgs)
                outputs = non_max_suppression(outputs, 80, conf_thres=self.config.conf_thres, nms_thres=self.config.nms_thres)

            for output, annotations in zip(outputs, targets):

                all_detections.append([np.array([]) for _ in range(num_classes)])
                if output is not None:
                    # Get predicted boxes, confidence scores and labels
                    pred_boxes = output[:, :5].cpu().numpy()
                    scores = output[:, 4].cpu().numpy()
                    pred_labels = output[:, -1].cpu().numpy()

                    # Order by confidence
                    sort_i = np.argsort(scores)
                    pred_labels = pred_labels[sort_i]
                    pred_boxes = pred_boxes[sort_i]

                    for label in range(num_classes):
                        all_detections[-1][label] = pred_boxes[pred_labels == label]

                all_annotations.append([np.array([]) for _ in range(num_classes)])
                if any(annotations[:, -1] > 0):

                    annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
                    _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                    # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                    annotation_boxes = np.empty_like(_annotation_boxes)
                    annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                    annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                    annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                    annotation_boxes *= 416

                    for label in range(num_classes):
                        all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

        average_precisions = {}
        for label in range(num_classes):
            true_positives = []
            scores = []
            num_annotations = 0

            for i in tqdm.tqdm(range(len(all_annotations)), desc=("Computing AP for class '{%s}'" %(label))):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]

                num_annotations += annotations.shape[0]
                detected_annotations = []

                for *bbox, score in detections:
                    scores.append(score)

                    if annotations.shape[0] == 0:
                        true_positives.append(0)
                        continue

                    overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= self.config.iou_thres and assigned_annotation not in detected_annotations:
                        true_positives.append(1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        true_positives.append(0)

            # no annotations -> AP for this class is 0
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            true_positives = np.array(true_positives)
            false_positives = np.ones_like(true_positives) - true_positives
            # sort by score
            indices = np.argsort(-np.array(scores))
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        print("Average Precisions:")
        for c, ap in average_precisions.items():
            print(" Class '{%s}' - AP: %f" % (c, ap))

        mAP = np.mean(list(average_precisions.values()))
        print("mAP: {%f}" % (mAP))
