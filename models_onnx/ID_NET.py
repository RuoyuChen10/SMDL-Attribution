import onnx_tf.backend
import onnx
import tensorflow as tf
import numpy as np

class ONNX_Face_Recognition:
    def __init__(self, model_path):
        self.model_path = model_path
        onnx_model = onnx.load(self.model_path)
        self.model = onnx_tf.backend.prepare(onnx_model, device='GPU:0')

    def __call__(self, inputs):
        assert len(inputs.shape) == 4
        # with tf.compat.v1.Session() as sess:
        #     inputs = sess.run(inputs)
        inputs = inputs.numpy()
        inputs = inputs.transpose(0, 3, 1, 2)
        # inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
        return self.model.run(inputs)[0]