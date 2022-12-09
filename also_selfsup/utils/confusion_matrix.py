import numpy as np
from typing import Dict, List, Tuple

class ConfusionMatrix:
    """
    Class for confusion matrix with various convenient methods.
    """
    def __init__(self, num_classes: int, ignore_idx: int = None):
        """
        Initialize a ConfusionMatrix object.
        :param num_classes: Number of classes in the confusion matrix.
        :param ignore_idx: Index of the class to be ignored in the confusion matrix.
        """
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx

        self.global_cm = None

    def update(self, gt_array: np.ndarray, pred_array: np.ndarray) -> None:
        """
        Updates the global confusion matrix.
        :param gt_array: An array containing the ground truth labels.
        :param pred_array: An array containing the predicted labels.
        """
        cm = self._get_confusion_matrix(gt_array, pred_array)

        if self.global_cm is None:
            self.global_cm = cm
        else:
            self.global_cm += cm

    def _get_confusion_matrix(self, gt_array: np.ndarray, pred_array: np.ndarray) -> np.ndarray:
        """
        Obtains the confusion matrix for the segmentation of a single point cloud.
        :param gt_array: An array containing the ground truth labels.
        :param pred_array: An array containing the predicted labels.
        :return: N x N array where N is the number of classes.
        """
        assert all((gt_array >= 0) & (gt_array < self.num_classes)), \
            "Error: Array for ground truth must be between 0 and {} (inclusive).".format(self.num_classes - 1)
        assert all((pred_array > 0) & (pred_array < self.num_classes)), \
            "Error: Array for predictions must be between 1 and {} (inclusive).".format(self.num_classes - 1)

        label = self.num_classes * gt_array.astype('int') + pred_array
        count = np.bincount(label, minlength=self.num_classes ** 2)

        # Make confusion matrix (rows = gt, cols = preds).
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)

        # For the class to be ignored, set both the row and column to 0 (adapted from
        # https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py).
        if self.ignore_idx is not None:
            confusion_matrix[self.ignore_idx, :] = 0
            confusion_matrix[:, self.ignore_idx] = 0

        return confusion_matrix

    def get_per_class_iou(self) -> List[float]:
        """
        Gets the IOU of each class in a confusion matrix.
        :return: An array in which the IOU of a particular class sits at the array index corresponding to the
                 class index.
        """
        conf = self.global_cm.copy()

        # Get the intersection for each class.
        intersection = np.diagonal(conf)

        # Get the union for each class.
        ground_truth_set = conf.sum(axis=1)
        predicted_set = conf.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        # Get the IOU for each class.
        # In case we get a division by 0, ignore / hide the error(adapted from
        # https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py).
        with np.errstate(divide='ignore', invalid='ignore'):
            iou_per_class = intersection / (union.astype(np.float32))

        return iou_per_class

    def get_mean_iou(self) -> float:
        """
        Gets the mean IOU (mIOU) over the classes.
        :return: mIOU over the classes.
        """
        iou_per_class = self.get_per_class_iou()
        miou = float(np.nanmean(iou_per_class))
        return miou

    def get_freqweighted_iou(self) -> float:
        """
        Gets the frequency-weighted IOU over the classes.
        :return: Frequency-weighted IOU over the classes.
        """
        conf = self.global_cm.copy()

        # Get the number of points per class (based on ground truth).
        num_points_per_class = conf.sum(axis=1)

        # Get the total number of points in the eval set.
        num_points_total = conf.sum()

        # Get the IOU per class.
        iou_per_class = self.get_per_class_iou()

        # Weight the IOU by frequency and sum across the classes.
        freqweighted_iou = float(np.nansum(num_points_per_class * iou_per_class) / num_points_total)

        return freqweighted_iou