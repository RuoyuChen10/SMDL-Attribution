import numpy as np
import onnx_tf.backend
import onnx

class AttributeModel:
    def __init__(self, model_path):
        self.model_path = model_path
        onnx_model = onnx.load(self.model_path)
        self.model = onnx_tf.backend.prepare(onnx_model, device='GPU:0')

        self.Face_attributes_name = [
            "Gender","Age","Race","Bald","Wavy Hair",
            "Receding Hairline","Bangs","Sideburns","Hair color","no beard",
            "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
            "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
            "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
            "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
        ]
        self.Gender = ["Male","Female"]
        self.Age = ["Young","Middle Aged","Senior"]
        self.Race = ["Asian","White","Black"]
        self.Hair_color = ["Black Hair","Blond Hair","Brown Hair","Gray Hair","Unknown Hair"]

        self.desired_attribute = [
            "Male","Female","Young","Middle Aged","Senior","Asian","White","Black","Bald","Wavy Hair",
            "Receding Hairline","Bangs","Sideburns","Black Hair","Blond Hair","Brown Hair","Gray Hair","no beard",
            "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
            "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
            "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
            "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
        ]# currently Sideburns not support

        self.default_attribute = [
            "Male","Female","Young","Middle Aged","Senior","Asian","White","Black","Bald","Wavy Hair",
            "Receding Hairline","Bangs","Sideburns","Black Hair","Blond Hair","Brown Hair","Gray Hair","no beard",
            "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
            "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
            "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
            "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
        ]

        # self.facial_attribute = ["Male", "Female", "Young", "Middle Aged", "Senior", "Asian", "White", "Black", "Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows","Big Lips","Big Nose","Pointy Nose"]

        self.facial_attribute = [
            "Male","Female","Young","Middle Aged","Senior","Asian","White","Black","Bald","Wavy Hair",
            "Receding Hairline","Bangs","Sideburns","Black Hair","Blond Hair","Brown Hair","Gray Hair","no beard",
            "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
            "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
            "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
            "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
        ]

        self.type = "VGGFace2"

    def softmax(self, f):
        f -= np.max(f)
        return np.exp(f) / np.exp(f).sum(axis=1, keepdims=True)

    def set_idx_list(self, attribute=["Male", "Young", "Middle Aged", "Senior", "Asian","White","Black"]):
        self.desired_attribute = attribute

    def set_idx(self, attr):
        self.desired_attribute += [attr]

    # def set_single_idx(self, attr):
    #     self.desired_attribute = attr
    def predict_attribute(self, inputs, threshold = 0.5):
        assert len(inputs.shape) == 4
        inputs = inputs.transpose(0, 3, 1, 2)

        out = np.array([[] for i in range(inputs.shape[0])])
        output = self.model.run(inputs)
        for attribute in self.facial_attribute:
            if attribute in self.Gender:
                out = np.append(out,
                    self.softmax(output[0])[:,self.Gender.index(attribute)][:, np.newaxis], 1
                    )
            elif attribute in self.Age:
                out = np.append(out,
                    self.softmax(output[1])[:,self.Age.index(attribute)][:, np.newaxis], 1
                    )
            elif attribute in self.Race:
                out = np.append(out,
                    self.softmax(output[2])[:,self.Race.index(attribute)][:, np.newaxis], 1
                    )
            elif attribute in self.Hair_color:
                out = np.append(out,
                    self.softmax(output[8])[:,self.Hair_color.index(attribute)][:, np.newaxis], 1
                    )
            else:
                out = np.append(out,
                    self.softmax(output[self.Face_attributes_name.index(attribute)])[:,0][:, np.newaxis], 1
                    )
                
        return (out > threshold).astype(int)


    def __call__(self, inputs):
        assert len(inputs.shape) == 4
        inputs = inputs.transpose(0, 3, 1, 2)

        out = np.array([[] for i in range(inputs.shape[0])])
        output = self.model.run(inputs)
        for attribute in self.desired_attribute:
            if attribute in self.Gender:
                out = np.append(out,
                    output[0][:,self.Gender.index(attribute)][:, np.newaxis], 1
                    )
            elif attribute in self.Age:
                out = np.append(out,
                    output[1][:,self.Age.index(attribute)][:, np.newaxis], 1
                    )
            elif attribute in self.Race:
                out = np.append(out,
                    output[2][:,self.Race.index(attribute)][:, np.newaxis], 1
                    )
            elif attribute in self.Hair_color:
                out = np.append(out,
                    output[8][:,self.Hair_color.index(attribute)][:, np.newaxis], 1
                    )
            else:
                out = np.append(out,
                    output[self.Face_attributes_name.index(attribute)][:,0][:, np.newaxis], 1
                    )
        
        return out