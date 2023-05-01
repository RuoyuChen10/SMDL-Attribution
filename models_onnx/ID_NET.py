import onnx_tf.backend
import onnx

class ONNX_Face_Recognition:
    def __init__(self, model_path):
        self.model_path = model_path
        onnx_model = onnx.load(self.model_path)
        self.model = onnx_tf.backend.prepare(onnx_model, device='GPU:1')

    def __call__(self, inputs):
        assert len(inputs.shape) == 4
        inputs = inputs.transpose(0, 3, 1, 2)
        return self.model.run(inputs)[0]