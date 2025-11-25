import torch


def confusion_matrix(preds, labels, num_classes):
    """
    preds, labels: 1D tensors of same length
    """
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=preds.device)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        if 0 <= t < num_classes:
            conf[t, p] += 1
    return conf


def iou_from_confmat(confmat):
    """
    confmat: (C, C)
    returns: per_class_iou (C,), miou
    """
    num_classes = confmat.shape[0]
    ious = []
    for c in range(num_classes):
        tp = confmat[c, c].float()
        fn = confmat[c, :].sum().float() - tp  # false negatives
        fp = confmat[:, c].sum().float() - tp  # false positives
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
        else:
            ious.append(torch.tensor(0.0, device=confmat.device))
    ious = torch.stack(ious)
    miou = ious.mean()
    return ious, miou
