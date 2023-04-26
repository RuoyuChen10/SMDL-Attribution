from xplique.attributions.global_sensitivity_analysis.gsa_attribution_method import GSABaseAttributionMethod
from xplique.types import Callable, Union, Optional, Tuple
from xplique.attributions.global_sensitivity_analysis.samplers import Sampler, TFSobolSequence
from xplique.attributions import *
from xplique.metrics import *

from xplique_addons import *

from enum import Enum

import cv2
import numpy as np
import tensorflow as tf

from xplique.commons import batch_tensor, repeat_labels
from xplique.attributions.base import BlackBoxExplainer, sanitize_input_output
from xplique.attributions.global_sensitivity_analysis.perturbations import amplitude, inpainting, blurring

class FaceHsicAttributionMethod(GSABaseAttributionMethod):
    """
    HSIC Attribution Method.
    Compute the dependance of each input dimension wrt the output using Hilbert-Schmidt Independance
    Criterion, a perturbation function on a grid and an adapted sampling as described in
    the original paper.

    Ref. Novello, Fel, Vigouroux, Making Sense of Dependance: Efficient Black-box Explanations
    Using Dependence Measure, https://arxiv.org/abs/2206.06219

    Parameters
    ----------
    model
        Model used for computing explanations.
    grid_size
        Cut the image in a grid of (grid_size, grid_size) to estimate an indice per cell.
    nb_design
        Number of design for the sampler.
    sampler
        Sampler used to generate the (quasi-)monte carlo samples, LHS or QMC.
        For more option, see the sampler module. Note that the original paper uses LHS but here
        the default sampler is TFSobolSequence as LHS requires scipy 1.7.0.
    estimator
        Estimator used to compute the HSIC score.
    perturbation_function
        Function to call to apply the perturbation on the input. Can also be string in
        'inpainting', 'blur'.
    batch_size
        Batch size to use for the forwards.
    """
    def __init__(
        self,
        model,
        grid_size: int = 8,
        nb_design: int = 500,
        sampler: Optional[Sampler] = None,
        estimator: Optional[HsicEstimator] = None,
        perturbation_function: Optional[Union[Callable, str]] = "inpainting",
        batch_size=256,
    ):

        sampler = sampler if sampler is not None else TFSobolSequence(binary=True)
        estimator = (
            estimator if estimator is not None else BinaryEstimator(output_kernel="rbf")
        )

        assert isinstance(sampler, Sampler), "The sampler must be a valid Sampler."
        assert isinstance(
            estimator, HsicEstimator
        ), "The estimator must be a valid HsicEstimator."

        if isinstance(estimator, BinaryEstimator):
            assert sampler.binary, "The sampler must be binary for BinaryEstimator."

        super().__init__(model = model, sampler = sampler, estimator = estimator,
                         grid_size = grid_size, nb_design = nb_design,
                         perturbation_function = perturbation_function, batch_size = batch_size,
        )
    @sanitize_input_output
    def explain(
        self,
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> tf.Tensor:
        """
        Compute the total Sobol' indices according to the explainer parameter (perturbation
        function, grid size...). Accept Tensor, numpy array or tf.data.Dataset (in that case
        targets is None).

        Parameters
        ----------
        inputs
            Images to be explained, either tf.dataset, Tensor or numpy array.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape (N, W, H, C) or (N, W, H).
        targets
            One-hot encoding for classification or direction {-1, +1} for regression.
            Tensor or numpy array.
            Expected shape (N, C) or (N).

        Returns
        -------
        attributions_maps
            GSA Attribution Method explanations, same shape as the inputs except for the channels.
        """
        # pylint: disable=E1101

        input_shape = (inputs.shape[1], inputs.shape[2])
        heatmaps = None

        for inp, target in zip(inputs, targets):

            perturbator = self.perturbation_function(inp)
            outputs = None

            for batch_masks in batch_tensor(self.masks, self.batch_size):

                batch_x, batch_y = self._batch_perturbations(
                    batch_masks, perturbator, target, input_shape
                )
                batch_outputs = self.inference_function(self.model, batch_x, batch_y)

                outputs = (
                    batch_outputs
                    if outputs is None
                    else tf.concat([outputs, batch_outputs], axis=0)
                )

            heatmap = self.estimator(self.masks, outputs, self.nb_design)
            heatmap = cv2.resize(heatmap, input_shape, interpolation=cv2.INTER_CUBIC)[
                None, :, :
            ]

            heatmaps = (
                heatmap if heatmaps is None else tf.concat([heatmaps, heatmap], axis=0)
            )

        return heatmaps

    @staticmethod
    @tf.function
    def _batch_perturbations(
        masks: tf.Tensor,
        perturbator: Callable,
        target: tf.Tensor,
        input_shape: Tuple[int, int],
    ) -> Union[tf.Tensor, tf.Tensor]:
        """
        Prepare perturbated input and replicated targets before a batch inference.

        Parameters
        ----------
        masks
            Perturbation masks in lower dimensions (grid_size, grid_size).
        perturbator
            Perturbation function to be called with the upsampled masks.
        target
            Label of a single prediction
        input_shape
            Shape of a single input

        Returns
        -------
        perturbated_inputs
            One inputs perturbated for each masks, according to the pertubation function
            modulated by the masks values.
        repeated_targets
            Replicated labels, one for each masks.
        """
        repeated_targets = repeat_labels(target[None, :], len(masks))

        upsampled_masks = tf.image.resize(masks, input_shape, method="nearest")
        perturbated_inputs = perturbator(upsampled_masks)

        return perturbated_inputs, repeated_targets