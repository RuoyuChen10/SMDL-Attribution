import numpy as np
import onnx_tf.backend
import onnx

class AttributeModel:
    def __init__(self, model_path):
        self.model_path = model_path
        onnx_model = onnx.load(self.model_path)
        self.model = onnx_tf.backend.prepare(onnx_model, device='GPU:0')

        self.Face_attributes_name = [
            'male', 'young', 
            'arched_eyebrows', 'bushy_eyebrows',
            'mouth_slightly_open', 'big_lips',
            'big_nose', 'pointy_nose',
            'bags_under_eyes', 'narrow_eyes'
        ]
        
        self.desired_attribute = [
            'male', 'female', 'young', 'old',
            'arched_eyebrows', 'bushy_eyebrows',
            'mouth_slightly_open', 'big_lips',
            'big_nose', 'pointy_nose',
            'bags_under_eyes', 'narrow_eyes'
        ]# currently Sideburns not support

        self.default_attribute = [
            'male', 'young',
            'arched_eyebrows', 'bushy_eyebrows',
            'mouth_slightly_open', 'big_lips',
            'big_nose', 'pointy_nose',
            'bags_under_eyes', 'narrow_eyes'
        ]

        self.facial_attribute = [
            'male', 'female', 'young', 'old',
            'arched_eyebrows', 'bushy_eyebrows',
            'mouth_slightly_open', 'big_lips',
            'big_nose', 'pointy_nose',
            'bags_under_eyes', 'narrow_eyes'
        ]

        self.type = "CelebA"

    def sigmoid(self, f):
        return 1/(1 + np.exp(-f))

    def set_idx_list(self, attribute=['male', 'young',
            'arched_eyebrows', 'bushy_eyebrows',
            'mouth_slightly_open', 'big_lips',
            'big_nose', 'pointy_nose',
            'bags_under_eyes', 'narrow_eyes']):
        self.desired_attribute = attribute

    def set_idx(self, attr):
        self.desired_attribute += [attr]

    def activation(self, input_, predict):
        if predict:
            return self.sigmoid(input_)
        else:
            return input_

    def __call__(self, inputs, predict = "False"):
        assert len(inputs.shape) == 4
        inputs = inputs.transpose(0, 3, 1, 2)

        out = np.array([[] for i in range(inputs.shape[0])])
        output = self.model.run(inputs)[0]
        for attribute in self.desired_attribute:
            if attribute == "female":
                out = np.append(out,
                    self.activation(
                        - output[:, 0][:, np.newaxis], 
                    predict), 1)
            elif attribute == "old":
                out = np.append(out,
                    self.activation(
                        - output[:, 1][:, np.newaxis], 
                    predict), 1)
            else:
                out = np.append(out,
                    self.activation(
                        output[:,self.Face_attributes_name.index(attribute)][:, np.newaxis], 
                    predict), 1)
        
        return out