import os
import cv2
import numpy as np
import random

import tensorflow as tf

from scipy.stats import pearsonr

from tqdm import tqdm

from tqdm.contrib import tzip

class SensitivityN(object):
    """
    Reference Paper:
        Marco Ancona, Enea Ceolini, Cengiz Öztireli, Markus Gross:
        Towards better understanding of gradient-based attribution methods for Deep Neural Networks. ICLR (Poster) 2018
    """
    def __init__(self,
                 model,
                 inputs,
                 targets,
                 n_percentage = 0.1,
                 grad_size = 10,
                 nb_samples = 100):
        super(SensitivityN, self).__init__()
        self.model = model
        self.inputs = inputs
        self.targets = targets

        self.grad_size = grad_size
        self.nb_samples = nb_samples

        # cardinal of subset (n in the equation)
        self.subset_size = int(grad_size ** 2 * n_percentage)   # mask size
    
        self.base_init()

    def base_init(self):
        x_index = []
        y_index = []
        for i in range(self.grad_size):
            x_index += [i for i_ in range(self.grad_size)]
            y_index += [i_ for i_ in range(self.grad_size)]

        self.x_index = np.array(x_index)
        self.y_index = np.array(y_index)
        self.idx_index = list(range(len(self.x_index)))

        self.zeros_template = np.zeros((self.grad_size, self.grad_size))
    
    def generate_mask(self):
        masks = []
        for i in range(self.nb_samples):
            idx = random.sample(self.idx_index, self.subset_size)
            mask = self.zeros_template.copy()
            mask[self.x_index[idx], self.y_index[idx]] = 1
            mask = cv2.resize(
                mask.astype(np.uint8), (self.inputs.shape[1], self.inputs.shape[2]), interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
        return np.array(masks)

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
        explanations = np.array(explanations)
        assert len(explanations) == len(self.inputs), "The number of explanations must be the " \
                                            f"same as the number of inputs: {len(explanations)}"\
                                            f" vs {len(self.inputs)}"
        
        correlations = []
        

        for inp, label, phi in tzip(self.inputs, self.targets, explanations):
            label = tf.repeat(label[None, :], self.nb_samples, 0)   # shape=(nb_samples, num_class)

            maskes = self.generate_mask()
            
            attribution_score = (np.array([phi]) * maskes).sum(axis=(1,2))
            
            # mask image
            masked_images = np.array([inp]) * (1 - maskes)[:,:,:,np.newaxis]
            input_conc = np.concatenate((np.array([inp]), masked_images), axis=0)
            
            res = self.model(input_conc)
            res_source = tf.repeat(tf.expand_dims(res[0],0), self.nb_samples, 0) * label
            res_baseline = res[1:] * label
            variation = (res_source - res_baseline).numpy().sum(-1)
            
            corr, p_value = pearsonr(attribution_score, variation)
            
            correlations.append(corr)

        return np.array(correlations).mean()