

import torch
from torch import nn
import numpy as np
# from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_recall_curve


from meddet.model.nnModules import ComponentModule


class AUC(ComponentModule):
    def __init__(self, aggregate=None):
        super().__init__()
        self.aggregate = aggregate
        assert aggregate in [None, 'mean', 'sum', 'none']
        self.aggregate = aggregate is None and 'none' or aggregate

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(aggregate={self.aggregate})'
        return repr_str
    
    def __call__(self,
                 net_output: torch.Tensor,
                 target: torch.Tensor):
        """

        Args:
            net_output: b,c,d,h,w
            target: b,c,d,h,w

        Returns:

        """
        assert net_output.ndim == target.ndim, f'Dimension not matched! {net_output.shape}, {target.shape}'

        num_classes = net_output.shape[1]
        auc = torch.zeros(num_classes).to(target.device)
        for i in range(num_classes):
            prediction_i = (net_output[:, i, ...]).cpu().numpy()
            target_i = ((target == i) * 1).cpu().numpy()
            if np.max(target_i) == 0:
                auc_i = 1.0 / (1.0 + np.sum(np.clip(prediction_i, 0, 1)))
            else:
                try:
                    auc_i = roc_auc_score(target_i.flatten(), prediction_i.flatten())
                except Exception as e:
                    print(e)
                    auc_i = 0.0
            auc[i] = torch.tensor(auc_i)

        if self.aggregate == 'none':
            metric = dict(zip([f"auc_class_{i}" for i in range(num_classes)], auc))
        elif self.aggregate == 'mean':
            metric = {'mean_auc': torch.mean(auc)}
        else:
            metric = {'sum_auc': torch.sum(auc)}
        return metric
    
    def plot_ROC(self):
        pass
        # fpr, tpr, thresholds = roc_curve((y_true), y_scores)
        # roc_curve = plt.figure()
        # plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
        # plt.title('ROC curve')
        # plt.xlabel("FPR (False Positive Rate)")
        # plt.ylabel("TPR (True Positive Rate)")
        # plt.legend(loc="lower right")
        # plt.savefig(path_experiment + "ROC.png")
        #
        # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
        # recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
        # AUC_prec_rec = np.trapz(precision, recall)
        # print
        # "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
        # prec_rec_curve = plt.figure()
        # plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
        # plt.title('Precision - Recall curve')
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.legend(loc="lower right")
        # plt.savefig(path_experiment + "Precision_recall.png")
        #
        # # Confusion matrix
        # threshold_confusion = 0.5
        # print
        # "\nConfusion matrix:  Custom threshold (for positive) of " + str(threshold_confusion)
        # y_pred = np.empty((y_scores.shape[0]))
        # for i in range(y_scores.shape[0]):
        #     if y_scores[i] >= threshold_confusion:
        #         y_pred[i] = 1
        #     else:
        #         y_pred[i] = 0
        # confusion = confusion_matrix(y_true, y_pred)
        # print
        # confusion