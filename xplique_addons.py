import tensorflow as tf
import tensorflow_probability as tfp

from xplique.attributions.global_sensitivity_analysis.perturbations import *
from xplique.attributions import *
from xplique.metrics import *
from xplique.attributions.global_sensitivity_analysis.samplers import *
from xplique.attributions.global_sensitivity_analysis.hsic_estimators import *
from xplique.types import Callable, Union, Optional, Tuple
from xplique.commons import batch_tensor, repeat_labels
from xplique.attributions.base import BlackBoxExplainer, sanitize_input_output
from xplique.attributions.global_sensitivity_analysis.perturbations import amplitude, inpainting, blurring
from xplique.attributions.global_sensitivity_analysis.sobol_attribution_method import *

import openturns as ot
import scipy
ot.ResourceMap.SetAsString("SobolIndicesExperiment-SamplingMethod", "QMC")

class HsicAttributionMethod(SobolAttributionMethod):


    def __init__(
        self,
        model,
        grid_size,
        nb_design,
        sampler,
        estimator,
        perturbation_function,
        batch_size=256
    ):

        BlackBoxExplainer.__init__(self, model, batch_size)

        self.grid_size = grid_size
        self.nb_design = nb_design

        if isinstance(perturbation_function, str):
            if perturbation_function == "inpainting":
                self.perturbation_function = inpainting
            else:
                self.perturbation_function = blurring
            # self.perturbation_function = PerturbationFunction.from_string(perturbation_function)
        else:
            self.perturbation_function = perturbation_function

        self.sampler = sampler 
        self.estimator = estimator

        self.masks = self.sampler(grid_size**2, nb_design).reshape((-1, grid_size, grid_size, 1))

class HsicSampler(ScipySampler):
    """
    Pure Monte-Carlo sampler for HSIC.
    """
    def __init__(self, binary = False):
      self.binary = binary

    def __call__(self, dimension, nb_design):
        masks = np.random.random((nb_design, int(dimension**0.5), int(dimension**0.5)))

        if self.binary:
          masks = np.round(masks)
        return np.array(masks, np.float32)

class HsicQMCSampler(ScipySampler):
    """
    Pure Monte-Carlo sampler for HSIC.
    """
    def __init__(self, binary = False):
      self.binary = binary

    def __call__(self, dimension, nb_design):
        distributions = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * dimension)
        design = ot.LowDiscrepancyExperiment(ot.SobolSequence(), distributions, nb_design, True)
            
        samples = design.generate()    
        masks = np.array(samples, dtype=np.float32)
        if self.binary:
          masks = np.round(masks)
        return masks

class HsicLHSSampler(ScipySampler):
    """
    Pure Monte-Carlo sampler for HSIC.
    """
    def __init__(self, binary = False):
      self.qmc = scipy.stats.qmc
      self.binary = binary

    def __call__(self, dimension, nb_design):
        # dimension 49, nb_design 1536
        sampler = self.qmc.LatinHypercube(dimension*2)
        masks = sampler.random(nb_design//2).astype(np.float32)
        if self.binary:
          masks= np.round(masks)  # (768, 98)
        return np.array(masks, np.float32)

class HsicEstimator(SobolEstimator):

    def __init__(self, sigmoid = False, standardize = False, kernel_type="rbf", base_inter=0):
      # if true, apply sigmoid to all model outputs before computing hsic
      self.sigmoid = sigmoid 
      # if true, standardize the ouputs before computing hsic
      self.standardize = standardize
      # if true, the masks are considered binary
      self.kernel_type = kernel_type
      self.base_inter = base_inter
      if standardize and sigmoid:
        raise NotImplementedError('Standardize and sigmoid are activated!')

    @staticmethod
    @tf.function
    def _test_stat_binary(masks, L, n, nb_dim):
      X = tf.transpose(masks)
      width_x = 0.5

      H = tf.eye(n) - tf.ones((n,n)) / n
      X1 = tf.reshape(X, (nb_dim, n, 1))
      X2 = tf.reshape(X, (nb_dim, 1, n))
      K = 1 - (X1 - X2)**2
      
      HK = tf.einsum("jk,ikl->ijl", H, K)
      HL = tf.einsum("jk,kl->jl", H, L)

      Kc = tf.einsum("ijk,kl->ijl", HK, H)
      Lc = tf.einsum("jk,kl->jl", HL, H)

      score = tf.math.reduce_sum(Kc * tf.transpose(Lc), axis=[1, 2]) / n

      return score

    @staticmethod
    @tf.function
    def _test_stat_binary_all(X, L, n, nb_dim):

      H = tf.eye(n) - tf.ones((n,n)) / n
      nb_dim2 = X.shape[1]
      X1 = tf.reshape(X, (nb_dim, nb_dim2, n, 1))
      X2 = tf.transpose(X1, [0,1,3,2])
      K = 0.5 - (X1 - X2)**2
      #K = tf.math.reduce_prod( 1 + K, axis=1)  - 1
      #KL = K * L
      #score = tf.math.reduce_sum(KL, axis=(1,2))/ (n)**2
       
      K = tf.math.reduce_prod( 1 + K, axis=1)
      HK = tf.einsum("jk,ikl->ijl", H, K)
      HL = tf.einsum("jk,kl->jl", H, L)

      Kc = tf.einsum("ijk,kl->ijl", HK, H)
      Lc = tf.einsum("jk,kl->jl", HL, H)

      score = tf.math.reduce_sum(Kc * tf.transpose(Lc), axis=[1, 2]) / n

      return score

    @staticmethod
    @tf.function
    def _test_stat_sobolev(X, masks, L, n, nb_dim):

      X = tf.cast(X, tf.float32)
      width_x = 0.5

      H = tf.eye(n) - tf.ones((n,n)) / n
      H = tf.cast(H, tf.float32)
      #X = tf.transpose(X, [0,2,1])
      X1 = tf.reshape(X, (nb_dim, nb_dim-1, n, 1))
      X2 = tf.reshape(X, (nb_dim, nb_dim-1, 1, n))
      #XX = tf.math.reduce_sum(tf.math.abs(X1 - X2), axis=1)
      XX = tf.math.abs(X1 - X2)
      B2XX = XX**2 - XX + 1/6
      Xr1 = tf.math.abs(X1) - 0.5
      Xr2 = tf.math.abs(X2) - 0.5
      B1XX = tf.einsum("ijkl, ijlm-> ijkm", Xr1, Xr2)
      K = B2XX / 2 + B1XX
      K = tf.math.reduce_prod( 1 + K, axis=1)
      
      HK = tf.einsum("jk,ikl->ijl", H, K)
      HL = tf.einsum("jk,kl->jl", H, L)
      Kc = tf.einsum("ijk,kl->ijl", HK, H)
      Lc = tf.einsum("jk,kl->jl", HL, H)

      score = tf.math.reduce_sum(Kc * tf.transpose(Lc), axis=[1, 2]) / n

      Xt = tf.reshape(masks, (nb_dim, n))
      Xt = tf.cast(Xt, tf.float32)
      X1 = tf.reshape(Xt, (nb_dim, nb_dim-1, n, 1))
      X2 = tf.reshape(Xt, (nb_dim, nb_dim-1, 1, n))
      #XX = tf.math.reduce_sum(tf.math.abs(X1 - X2), axis=1)
      XX = tf.math.abs(X1 - X2)
      B2XX = XX**2 - XX + 1/6
      Xr1 = tf.math.abs(X1) - 0.5
      Xr2 = tf.math.abs(X2) - 0.5
      B1XX = tf.einsum("ijkl, ijlm-> ijkm", Xr1, Xr2)
      K = B2XX / 2 + B1XX
      K = tf.math.reduce_prod( 1 + K, axis=1)
      HK = tf.einsum("jk,kl->jl", H, K)
      HL = tf.einsum("jk,kl->jl", H, L)
      Kc = tf.einsum("jk,kl->jl", HK, H)
      Lc = tf.einsum("jk,kl->jl", HL, H)
      scoretot = tf.math.reduce_sum(Kc * tf.transpose(Lc), axis=[0, 1]) / n
      return  1 - score / scoretot


    @staticmethod
    @tf.function
    def _test_stat(masks, L, n, nb_dim):
      X = tf.transpose(masks)
      width_x = 0.5

      H = tf.eye(n) - tf.ones((n,n)) / n
      X1 = tf.reshape(X, (nb_dim, n, 1))
      X2 = tf.reshape(X, (nb_dim, 1, n))
      K = (X1 - X2)**2
      K = tf.math.exp(-K/2/(width_x**2))
      
      HK = tf.einsum("jk,ikl->ijl", H, K)
      HL = tf.einsum("jk,kl->jl", H, L)

      Kc = tf.einsum("ijk,kl->ijl", HK, H)
      Lc = tf.einsum("jk,kl->jl", HL, H)

      score = tf.math.reduce_sum(Kc * tf.transpose(Lc), axis=[1, 2]) / n

      
      return score

    @staticmethod
    @tf.function
    def _rbf_dot_1d(X, Y, deg):
      XY = (X - tf.transpose(Y))**2
      H = tf.exp(-XY/2/(deg**2))
      return H

    def __call__(self, masks, outputs, nb_design):
      # masks shape (1536, 7, 7, 1)
      # outputs shape 1536
      nb_dim = self.masks_dim(masks)
      dimension = int(nb_dim**0.5)
      n = masks.shape[0]

      outputs = tf.cast(outputs, tf.float32)  # shape 1536 vector.
      if self.sigmoid:
        outputs = tf.nn.sigmoid(outputs)
      
      if self.standardize:
        mu_outputs = tf.reduce_mean(outputs)
        std_outputs = tf.math.reduce_std(outputs) + 1e-5
        outputs = (outputs - mu_outputs ) / std_outputs

      Y = tf.reshape(outputs, (n, 1))
      width_y = tfp.stats.percentile(Y, 50.)
      L = self._rbf_dot_1d(Y, Y, width_y)
      
      if self.kernel_type == "binary":
        scores = self._test_stat_binary(masks, L, n, nb_dim)
      elif self.kernel_type == "rbf":
        scores = self._test_stat(masks, L, n, nb_dim)
      elif self.kernel_type == "sobolev":
        X = np.reshape(masks, (n, nb_dim))
        X = np.transpose(X)
        X = np.array([X[np.where([j != i for j in range(nb_dim)])[0], :] for i in range(nb_dim)])
        L = tf.cast(L, tf.float32)

        scores = self._test_stat_sobolev(X, masks, L, n, nb_dim)
        scores = tf.reshape(scores, masks.shape[1:])
        scores = tf.transpose(scores, [1,0,2])
      elif self.kernel_type == "inter":
        X = np.reshape(masks, (n, nb_dim))
        X = np.transpose(X)
        X = np.array([X[[self.base_inter, i], :] for i in range(nb_dim)])
        L = tf.cast(L, tf.float32)
        scores = np.array(self._test_stat_binary_all(X, L, n, nb_dim))
        
        X = tf.transpose(masks)
        X = np.reshape(X, (nb_dim, 1, n, 1))
        scores_ind = np.array(self._test_stat_binary_all(X, L, n, nb_dim))
        
        X = tf.transpose(masks)
        X = np.reshape(X, (1, nb_dim, n, 1))
        scoretot = self._test_stat_binary_all(X, L, n, nb_dim)

        for i in range(scores.shape[0]):
            scores[i] = (scores[i]  - scores_ind[i] - scores_ind[self.base_inter])
        scores[self.base_inter] = np.max(scores)
        scores = tf.reshape(scores, masks.shape[1:])
        scores = tf.transpose(scores, [1,0,2])
      else:
        print("please specify a valid kernel")

      scores = tf.reshape(scores, masks.shape[1:])
      scores = tf.transpose(scores, [1,0,2])
      
      return np.array(scores, np.float32)