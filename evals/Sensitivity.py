import os
import cv2
import numpy as np

from scipy.stats import pearsonr

from tqdm import tqdm

class SensitivityN(object):
    def __init__(self,
                 model,
                 inputs,
                 targets,
                 n_percentage = 0.2,
                 grad_szie = 10,
                 nb_samples = 100):
        super(SensitivityN, self).__init__()
        self.model = model
        self.inputs = inputs
        self.targets = targets

        self.nb_samples = nb_samples

    def __call__(self, explanations):
        """
        Evaluate the Sensitivity-N score.

        Parameters
        ----------
        explanations
            Explanation for the inputs, labels to evaluate.

        Returns
        -------
        fidelity_score
        """