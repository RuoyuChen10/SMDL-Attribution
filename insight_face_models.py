import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def print_buildin_models():
    print(
        """
    >>>> buildin_models
    MXNet version resnet: mobilenet_m1, r18, r34, r50, r100, r101, se_r34, se_r50, se_r100
    Keras application: mobilenet, mobilenetv2, resnet50, resnet50v2, resnet101, resnet101v2, resnet152, resnet152v2
    EfficientNet: efficientnetb[0-7], efficientnetl2, efficientnetv2b[1-3], efficientnetv2[t, s, m, l, xl]
    Custom 1: ghostnet, mobilefacenet, mobilenetv3_small, mobilenetv3_large, se_mobilefacenet, se_resnext, vargface
    Or other names from keras.applications like DenseNet121 / InceptionV3 / NASNetMobile / VGG19.
    """,
        end="",
    )


def __init_model_from_name__(name, input_shape=(112, 112, 3), weights="imagenet", **kwargs):
    name_lower = name.lower()
    """ Basic model """
    if name_lower == "mobilenet":
        xx = keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights=weights, **kwargs)
    elif name_lower == "mobilenet_m1":
        from backbones import mobilenet_m1

        xx = mobilenet_m1.MobileNet(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name_lower == "mobilenetv2":
        xx = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights, **kwargs)
    elif "r18" in name_lower or "r34" in name_lower or "r50" in name_lower or "r100" in name_lower or "r101" in name_lower:
        from backbones import resnet  # MXNet insightface version resnet

        use_se = True if name_lower.startswith("se_") else False
        model_name = "ResNet" + name_lower[4:] if use_se else "ResNet" + name_lower[1:]
        use_se = kwargs.pop("use_se", use_se)
        model_class = getattr(resnet, model_name)
        xx = model_class(input_shape=input_shape, classes=0, use_se=use_se, model_name=model_name, **kwargs)
    elif name_lower.startswith("resnet"):  # keras.applications.ResNetxxx
        if name_lower.endswith("v2"):
            model_name = "ResNet" + name_lower[len("resnet") : -2] + "V2"
        else:
            model_name = "ResNet" + name_lower[len("resnet") :]
        model_class = getattr(keras.applications, model_name)
        xx = model_class(weights=weights, include_top=False, input_shape=input_shape, **kwargs)
    elif name_lower.startswith("efficientnetv2"):
        import keras_efficientnet_v2

        model_name = "EfficientNetV2" + name_lower[len("EfficientNetV2") :].upper()
        model_class = getattr(keras_efficientnet_v2, model_name)
        xx = model_class(pretrained=weights, num_classes=0, input_shape=input_shape, **kwargs)
    elif name_lower.startswith("efficientnet"):
        import keras_efficientnet_v2

        model_name = "EfficientNetV1" + name_lower[-2:].upper()
        model_class = getattr(keras_efficientnet_v2, model_name)
        xx = model_class(pretrained=weights, num_classes=0, input_shape=input_shape, **kwargs)
    elif name_lower.startswith("se_resnext"):
        from keras_squeeze_excite_network import se_resnext

        if name_lower.endswith("101"):  # se_resnext101
            depth = [3, 4, 23, 3]
        else:  # se_resnext50
            depth = [3, 4, 6, 3]
        xx = se_resnext.SEResNextImageNet(weights=weights, input_shape=input_shape, include_top=False, depth=depth)
    elif name_lower.startswith("mobilenetv3"):
        model_class = keras.applications.MobileNetV3Small if "small" in name_lower else keras.applications.MobileNetV3Large
        xx = model_class(input_shape=input_shape, include_top=False, weights=weights, include_preprocessing=False)
    elif "mobilefacenet" in name_lower or "mobile_facenet" in name_lower:
        from backbones import mobile_facenet

        use_se = True if "se" in name_lower else False
        xx = mobile_facenet.MobileFaceNet(input_shape=input_shape, include_top=False, name=name, use_se=use_se)
    elif name_lower == "ghostnet":
        from backbones import ghost_model

        xx = ghost_model.GhostNet(input_shape=input_shape, include_top=False, width=1.3, **kwargs)
    elif name_lower == "vargface":
        from backbones import vargface

        xx = vargface.VargFace(input_shape=input_shape, **kwargs)
    elif hasattr(keras.applications, name):
        model_class = getattr(keras.applications, name)
        xx = model_class(weights=weights, include_top=False, input_shape=input_shape, **kwargs)
    else:
        return None
    xx.trainable = True
    return xx


# MXNET: bn_momentum=0.9, bn_epsilon=2e-5, TF default: bn_momentum=0.99, bn_epsilon=0.001, PyTorch default: momentum=0.1, eps=1e-05
# MXNET: use_bias=True, scale=False, cavaface.pytorch: use_bias=False, scale=True
def buildin_models(
    stem_model,
    dropout=1,
    emb_shape=512,
    input_shape=(112, 112, 3),
    output_layer="GDC",
    bn_momentum=0.99,
    bn_epsilon=0.001,
    add_pointwise_conv=False,
    pointwise_conv_act="relu",
    use_bias=False,
    scale=True,
    weights="imagenet",
    **kwargs
):
    if isinstance(stem_model, str):
        xx = __init_model_from_name__(stem_model, input_shape, weights, **kwargs)
        name = stem_model
    else:
        name = stem_model.name
        xx = stem_model

    if bn_momentum != 0.99 or bn_epsilon != 0.001:
        print(">>>> Change BatchNormalization momentum and epsilon default value.")
        for ii in xx.layers:
            if isinstance(ii, keras.layers.BatchNormalization):
                ii.momentum, ii.epsilon = bn_momentum, bn_epsilon
        xx = keras.models.clone_model(xx)

    inputs = xx.inputs[0]
    nn = xx.outputs[0]

    if add_pointwise_conv:  # Model using `pointwise_conv + GDC` / `pointwise_conv + E` is smaller than `E`
        filters = nn.shape[-1] // 2 if add_pointwise_conv == -1 else 512  # Compitable with previous models...
        nn = keras.layers.Conv2D(filters, 1, use_bias=False, padding="valid", name="pw_conv")(nn)
        # nn = keras.layers.Conv2D(nn.shape[-1] // 2, 1, use_bias=False, padding="valid", name="pw_conv")(nn)
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="pw_bn")(nn)
        if pointwise_conv_act.lower() == "prelu":
            nn = keras.layers.PReLU(shared_axes=[1, 2], name="pw_" + pointwise_conv_act)(nn)
        else:
            nn = keras.layers.Activation(pointwise_conv_act, name="pw_" + pointwise_conv_act)(nn)

    if output_layer == "E":
        """Fully Connected"""
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="E_batchnorm")(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Flatten(name="E_flatten")(nn)
        nn = keras.layers.Dense(emb_shape, use_bias=use_bias, kernel_initializer="glorot_normal", name="E_dense")(nn)
    elif output_layer == "GAP":
        """GlobalAveragePooling2D"""
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="GAP_batchnorm")(nn)
        nn = keras.layers.GlobalAveragePooling2D(name="GAP_pool")(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Dense(emb_shape, use_bias=use_bias, kernel_initializer="glorot_normal", name="GAP_dense")(nn)
    elif output_layer == "GDC":
        """GDC"""
        nn = keras.layers.DepthwiseConv2D(nn.shape[1], use_bias=False, name="GDC_dw")(nn)
        # nn = keras.layers.Conv2D(nn.shape[-1], nn.shape[1], use_bias=False, padding="valid", groups=nn.shape[-1])(nn)
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="GDC_batchnorm")(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Conv2D(emb_shape, 1, use_bias=use_bias, kernel_initializer="glorot_normal", name="GDC_conv")(nn)
        nn = keras.layers.Flatten(name="GDC_flatten")(nn)
        # nn = keras.layers.Dense(emb_shape, activation=None, use_bias=use_bias, kernel_initializer="glorot_normal", name="GDC_dense")(nn)
    elif output_layer == "F":
        """F, E without first BatchNormalization"""
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Flatten(name="F_flatten")(nn)
        nn = keras.layers.Dense(emb_shape, use_bias=use_bias, kernel_initializer="glorot_normal", name="F_dense")(nn)

    # `fix_gamma=True` in MXNet means `scale=False` in Keras
    embedding = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, scale=scale, name="pre_embedding")(nn)
    embedding_fp32 = keras.layers.Activation("linear", dtype="float32", name="embedding")(embedding)

    basic_model = keras.models.Model(inputs, embedding_fp32, name=xx.name)
    
    return basic_model


@keras.utils.register_keras_serializable(package="keras_insightface")
class NormDense(keras.layers.Layer):
    def __init__(self, units=1000, kernel_regularizer=None, loss_top_k=1, append_norm=False, partial_fc_split=0, **kwargs):
        super(NormDense, self).__init__(**kwargs)
        # self.init = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
        self.init = keras.initializers.glorot_normal()
        # self.init = keras.initializers.TruncatedNormal(mean=0, stddev=0.01)
        self.units, self.loss_top_k, self.append_norm, self.partial_fc_split = units, loss_top_k, append_norm, partial_fc_split
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.supports_masking = False

    def build(self, input_shape):
        if self.partial_fc_split > 1:
            self.cur_id = self.add_weight(name="cur_id", shape=(), initializer="zeros", dtype="int64", trainable=False)
            self.sub_weights = self.add_weight(
                name="norm_dense_w_subs",
                shape=(self.partial_fc_split, input_shape[-1], self.units * self.loss_top_k),
                initializer=self.init,
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
        else:
            self.w = self.add_weight(
                name="norm_dense_w",
                shape=(input_shape[-1], self.units * self.loss_top_k),
                initializer=self.init,
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
        super(NormDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # tf.print("tf.reduce_mean(self.w):", tf.reduce_mean(self.w))
        if self.partial_fc_split > 1:
            # self.sub_weights.scatter_nd_update([[(self.cur_id - 1) % self.partial_fc_split]], [self.w])
            # self.w.assign(tf.gather(self.sub_weights, self.cur_id))
            self.w = tf.gather(self.sub_weights, self.cur_id)
            self.cur_id.assign((self.cur_id + 1) % self.partial_fc_split)

        norm_w = tf.nn.l2_normalize(self.w, axis=0, epsilon=1e-5)
        norm_inputs = tf.nn.l2_normalize(inputs, axis=1, epsilon=1e-5)
        output = K.dot(norm_inputs, norm_w)
        if self.loss_top_k > 1:
            output = K.reshape(output, (-1, self.units, self.loss_top_k))
            output = K.max(output, axis=2)
        if self.append_norm:
            # Keep norm value low by * -1, so will not affect accuracy metrics.
            output = tf.concat([output, tf.norm(inputs, axis=1, keepdims=True) * -1], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(NormDense, self).get_config()
        config.update(
            {
                "units": self.units,
                "loss_top_k": self.loss_top_k,
                "append_norm": self.append_norm,
                "partial_fc_split": self.partial_fc_split,
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package="keras_insightface")
class NormDenseVPL(NormDense):
    def __init__(self, batch_size, units=1000, kernel_regularizer=None, vpl_lambda=0.15, start_iters=8000, allowed_delta=200, **kwargs):
        super().__init__(units, kernel_regularizer, **kwargs)
        self.vpl_lambda, self.batch_size = vpl_lambda, batch_size  # Need the actual batch_size here, for storing inputs
        # self.start_iters, self.allowed_delta = 8000 * 128 // batch_size, 200 * 128 // batch_size # adjust according to batch_size
        self.start_iters, self.allowed_delta = start_iters, allowed_delta
        # print(">>>> [NormDenseVPL], vpl_lambda={}, start_iters={}, allowed_delta={}".format(vpl_lambda, start_iters, allowed_delta))

    def build(self, input_shape):
        # self.queue_features in same shape format as self.norm_features, for easier calling tf.tensor_scatter_nd_update
        self.norm_features = self.add_weight(name="norm_features", shape=(self.batch_size, input_shape[-1]), dtype=self.compute_dtype, trainable=False)
        self.queue_features = self.add_weight(name="queue_features", shape=(self.units, input_shape[-1]), initializer=self.init, trainable=False)
        self.queue_iters = self.add_weight(name="queue_iters", shape=(self.units,), initializer="zeros", dtype="int64", trainable=False)
        self.zero_queue_lambda = tf.zeros((self.units,), dtype=self.compute_dtype)
        self.iters = self.add_weight(name="iters", shape=(), initializer="zeros", dtype="int64", trainable=False)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # tf.print("tf.reduce_mean(self.w):", tf.reduce_mean(self.w))
        self.iters.assign_add(1)
        queue_lambda = tf.cond(
            self.iters > self.start_iters,
            lambda: tf.where(self.iters - self.queue_iters <= self.allowed_delta, self.vpl_lambda, 0.0),  # prepare_queue_lambda
            lambda: self.zero_queue_lambda,
        )
        tf.print(" - vpl_sample_ratio:", tf.reduce_mean(tf.cast(queue_lambda > 0, "float32")), end="")
        # self.queue_lambda = queue_lambda

        if self.partial_fc_split > 1:
            self.w = tf.gather(self.sub_weights, self.cur_id)
            self.cur_id.assign((self.cur_id + 1) % self.partial_fc_split)

        norm_w = K.l2_normalize(self.w, axis=0)
        injected_weight = norm_w * (1 - queue_lambda) + tf.transpose(self.queue_features) * queue_lambda
        injected_norm_weight = K.l2_normalize(injected_weight, axis=0)

        # set_queue needs actual input labels, it's done in callback VPLUpdateQueue

        norm_inputs = K.l2_normalize(inputs, axis=1)
        self.norm_features.assign(norm_inputs)
        output = K.dot(norm_inputs, injected_norm_weight)
        if self.append_norm:
            # Keep norm value low by * -1, so will not affect accuracy metrics.
            output = tf.concat([output, tf.norm(inputs, axis=1, keepdims=True) * -1], axis=-1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"batch_size": self.batch_size, "vpl_lambda": self.vpl_lambda})
        return config


def add_l2_regularizer_2_model(model, weight_decay, custom_objects={}, apply_to_batch_normal=False, apply_to_bias=False):
    # https://github.com/keras-team/keras/issues/2717#issuecomment-456254176
    if 0:
        regularizers_type = {}
        for layer in model.layers:
            rrs = [kk for kk in layer.__dict__.keys() if "regularizer" in kk and not kk.startswith("_")]
            if len(rrs) != 0:
                # print(layer.name, layer.__class__.__name__, rrs)
                if layer.__class__.__name__ not in regularizers_type:
                    regularizers_type[layer.__class__.__name__] = rrs
        print(regularizers_type)

    for layer in model.layers:
        attrs = []
        if isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.Conv2D):
            # print(">>>> Dense or Conv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["kernel_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.DepthwiseConv2D):
            # print(">>>> DepthwiseConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["depthwise_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif isinstance(layer, keras.layers.SeparableConv2D):
            # print(">>>> SeparableConv2D", layer.name, "use_bias:", layer.use_bias)
            attrs = ["pointwise_regularizer", "depthwise_regularizer"]
            if apply_to_bias and layer.use_bias:
                attrs.append("bias_regularizer")
        elif apply_to_batch_normal and isinstance(layer, keras.layers.BatchNormalization):
            # print(">>>> BatchNormalization", layer.name, "scale:", layer.scale, ", center:", layer.center)
            if layer.center:
                attrs.append("beta_regularizer")
            if layer.scale:
                attrs.append("gamma_regularizer")
        elif apply_to_batch_normal and isinstance(layer, keras.layers.PReLU):
            # print(">>>> PReLU", layer.name)
            attrs = ["alpha_regularizer"]

        for attr in attrs:
            if hasattr(layer, attr) and layer.trainable:
                setattr(layer, attr, keras.regularizers.L2(weight_decay / 2))

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    # temp_weight_file = "tmp_weights.h5"
    # model.save_weights(temp_weight_file)
    # out_model = keras.models.model_from_json(model.to_json(), custom_objects=custom_objects)
    # out_model.load_weights(temp_weight_file, by_name=True)
    # os.remove(temp_weight_file)
    # return out_model
    return keras.models.clone_model(model)


def replace_ReLU_with_PReLU(model, target_activation="PReLU", **kwargs):
    from tensorflow.keras.layers import ReLU, PReLU, Activation

    def convert_ReLU(layer):
        # print(layer.name)
        if isinstance(layer, ReLU) or (isinstance(layer, Activation) and layer.activation == keras.activations.relu):
            if target_activation == "PReLU":
                layer_name = layer.name.replace("_relu", "_prelu")
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                # Default initial value in mxnet and pytorch is 0.25
                return PReLU(shared_axes=[1, 2], alpha_initializer=tf.initializers.Constant(0.25), name=layer_name, **kwargs)
            elif isinstance(target_activation, str):
                layer_name = layer.name.replace("_relu", "_" + target_activation)
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                return Activation(activation=target_activation, name=layer_name, **kwargs)
            else:
                act_class_name = target_activation.__name__
                layer_name = layer.name.replace("_relu", "_" + act_class_name)
                print(">>>> Convert ReLU:", layer.name, "-->", layer_name)
                return target_activation(**kwargs)
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=convert_ReLU)


@keras.utils.register_keras_serializable(package="keras_insightface")
class AconC(keras.layers.Layer):
    """
    - [Github nmaac/acon](https://github.com/nmaac/acon/blob/main/acon.py)
    - [Activate or Not: Learning Customized Activation, CVPR 2021](https://arxiv.org/pdf/2009.04759.pdf)
    """

    def __init__(self, p1=1, p2=0, beta=1, **kwargs):
        super(AconC, self).__init__(**kwargs)
        self.p1_init = tf.initializers.Constant(p1)
        self.p2_init = tf.initializers.Constant(p2)
        self.beta_init = tf.initializers.Constant(beta)
        self.supports_masking = False

    def build(self, input_shape):
        self.p1 = self.add_weight(name="p1", shape=(1, 1, 1, input_shape[-1]), initializer=self.p1_init, trainable=True)
        self.p2 = self.add_weight(name="p2", shape=(1, 1, 1, input_shape[-1]), initializer=self.p2_init, trainable=True)
        self.beta = self.add_weight(name="beta", shape=(1, 1, 1, input_shape[-1]), initializer=self.beta_init, trainable=True)
        super(AconC, self).build(input_shape)

    def call(self, inputs, **kwargs):
        p1 = inputs * self.p1
        p2 = inputs * self.p2
        beta = inputs * self.beta
        return p1 * tf.nn.sigmoid(beta) + p2

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(AconC, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SAMModel(tf.keras.models.Model):
    """
    Arxiv article: [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/pdf/2010.01412.pdf)
    Implementation by: [Keras SAM (Sharpness-Aware Minimization)](https://qiita.com/T-STAR/items/8c3afe3a116a8fc08429)

    Usage is same with `keras.modeols.Model`: `model = SAMModel(inputs, outputs, rho=sam_rho, name=name)`
    """

    def __init__(self, *args, rho=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = tf.constant(rho, dtype=tf.float32)

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # 1st step
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        norm = tf.linalg.global_norm(gradients)
        scale = self.rho / (norm + 1e-12)
        e_w_list = []
        for v, grad in zip(trainable_vars, gradients):
            e_w = grad * scale
            v.assign_add(e_w)
            e_w_list.append(e_w)

        # 2nd step
        with tf.GradientTape() as tape:
            y_pred_adv = self(x, training=True)
            loss_adv = self.compiled_loss(y, y_pred_adv, sample_weight=sample_weight, regularization_losses=self.losses)
        gradients_adv = tape.gradient(loss_adv, trainable_vars)
        for v, e_w in zip(trainable_vars, e_w_list):
            v.assign_sub(e_w)

        # optimize
        self.optimizer.apply_gradients(zip(gradients_adv, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


def replace_add_with_stochastic_depth(model, survivals=(1, 0.8)):
    """
    - [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)
    - [tfa.layers.StochasticDepth](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/StochasticDepth)
    """
    from tensorflow_addons.layers import StochasticDepth

    add_layers = [ii.name for ii in model.layers if isinstance(ii, keras.layers.Add)]
    total_adds = len(add_layers)
    if isinstance(survivals, float):
        survivals = [survivals] * total_adds
    elif isinstance(survivals, (list, tuple)) and len(survivals) == 2:
        start, end = survivals
        survivals = [start - (1 - end) * float(ii) / total_adds for ii in range(total_adds)]
    survivals_dict = dict(zip(add_layers, survivals))

    def __replace_add_with_stochastic_depth__(layer):
        if isinstance(layer, keras.layers.Add):
            layer_name = layer.name
            new_layer_name = layer_name.replace("_add", "_stochastic_depth")
            new_layer_name = layer_name.replace("add_", "stochastic_depth_")
            survival_probability = survivals_dict[layer_name]
            if survival_probability < 1:
                print("Converting:", layer_name, "-->", new_layer_name, ", survival_probability:", survival_probability)
                return StochasticDepth(survival_probability, name=new_layer_name)
            else:
                return layer
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=__replace_add_with_stochastic_depth__)


def replace_stochastic_depth_with_add(model, drop_survival=False):
    from tensorflow_addons.layers import StochasticDepth

    def __replace_stochastic_depth_with_add__(layer):
        if isinstance(layer, StochasticDepth):
            layer_name = layer.name
            new_layer_name = layer_name.replace("_stochastic_depth", "_lambda")
            survival = layer.survival_probability
            print("Converting:", layer_name, "-->", new_layer_name, ", survival_probability:", survival)
            if drop_survival or not survival < 1:
                return keras.layers.Add(name=new_layer_name)
            else:
                return keras.layers.Lambda(lambda xx: xx[0] + xx[1] * survival, name=new_layer_name)
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=__replace_stochastic_depth_with_add__)


def convert_to_mixed_float16(model, convert_batch_norm=False):
    policy = keras.mixed_precision.Policy("mixed_float16")
    policy_config = keras.utils.serialize_keras_object(policy)
    from tensorflow.keras.layers import InputLayer, Activation
    from tensorflow.keras.activations import linear, softmax

    def do_convert_to_mixed_float16(layer):
        if not convert_batch_norm and isinstance(layer, keras.layers.BatchNormalization):
            return layer
        if isinstance(layer, InputLayer):
            return layer
        if isinstance(layer, NormDense):
            return layer
        if isinstance(layer, Activation) and layer.activation == softmax:
            return layer
        if isinstance(layer, Activation) and layer.activation == linear:
            return layer

        aa = layer.get_config()
        aa.update({"dtype": policy_config})
        bb = layer.__class__.from_config(aa)
        bb.build(layer.input_shape)
        bb.set_weights(layer.get_weights())
        return bb

    input_tensors = keras.layers.Input(model.input_shape[1:])
    mm = keras.models.clone_model(model, input_tensors=input_tensors, clone_function=do_convert_to_mixed_float16)
    if model.built:
        mm.compile(optimizer=model.optimizer, loss=model.compiled_loss, metrics=model.compiled_metrics)
        # mm.optimizer, mm.compiled_loss, mm.compiled_metrics = model.optimizer, model.compiled_loss, model.compiled_metrics
        # mm.built = True
    return mm


def convert_mixed_float16_to_float32(model):
    from tensorflow.keras.layers import InputLayer, Activation
    from tensorflow.keras.activations import linear

    def do_convert_to_mixed_float16(layer):
        if not isinstance(layer, InputLayer) and not (isinstance(layer, Activation) and layer.activation == linear):
            aa = layer.get_config()
            aa.update({"dtype": "float32"})
            bb = layer.__class__.from_config(aa)
            bb.build(layer.input_shape)
            bb.set_weights(layer.get_weights())
            return bb
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=do_convert_to_mixed_float16)


def convert_to_batch_renorm(model):
    def do_convert_to_batch_renorm(layer):
        if isinstance(layer, keras.layers.BatchNormalization):
            aa = layer.get_config()
            aa.update({"renorm": True, "renorm_clipping": {}, "renorm_momentum": aa["momentum"]})
            bb = layer.__class__.from_config(aa)
            bb.build(layer.input_shape)
            bb.set_weights(layer.get_weights() + bb.get_weights()[-3:])
            return bb
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=do_convert_to_batch_renorm)








# margin_softmax class wrapper
@keras.utils.register_keras_serializable(package="keras_insightface")
class MarginSoftmax(tf.keras.losses.Loss):
    def __init__(self, power=2, scale=0.4, scale_all=1.0, from_logits=False, label_smoothing=0, **kwargs):
        super(MarginSoftmax, self).__init__(**kwargs)
        self.power, self.scale, self.from_logits, self.label_smoothing = power, scale, from_logits, label_smoothing
        self.scale_all = scale_all
        if power != 1 and scale == 0:
            self.logits_reduction_func = lambda xx: xx**power
        elif power == 1 and scale != 0:
            self.logits_reduction_func = lambda xx: xx * scale
        else:
            self.logits_reduction_func = lambda xx: (xx**power + xx * scale) / 2

    def call(self, y_true, y_pred):
        # margin_soft = tf.where(tf.cast(y_true, dtype=tf.bool), (y_pred ** self.power + y_pred * self.scale) / 2, y_pred)
        margin_soft = tf.where(tf.cast(y_true, dtype=tf.bool), self.logits_reduction_func(y_pred), y_pred) * self.scale_all
        return tf.keras.losses.categorical_crossentropy(y_true, margin_soft, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super(MarginSoftmax, self).get_config()
        config.update(
            {
                "power": self.power,
                "scale": self.scale,
                "scale_all": self.scale_all,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ArcfaceLoss class
@keras.utils.register_keras_serializable(package="keras_insightface")
class ArcfaceLoss(tf.keras.losses.Loss):
    def __init__(self, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        # reduction = tf.keras.losses.Reduction.NONE if tf.distribute.has_strategy() else tf.keras.losses.Reduction.AUTO
        # super(ArcfaceLoss, self).__init__(**kwargs, reduction=reduction)
        super(ArcfaceLoss, self).__init__(**kwargs)
        self.margin1, self.margin2, self.margin3, self.scale = margin1, margin2, margin3, scale
        self.from_logits, self.label_smoothing = from_logits, label_smoothing
        self.threshold = np.cos((np.pi - margin2) / margin1)  # grad(theta) == 0
        self.theta_min = (-1 - margin3) * 2
        self.batch_labels_back_up = None
        # self.reduction_func = tf.keras.losses.CategoricalCrossentropy(
        #     from_logits=from_logits, label_smoothing=label_smoothing, reduction=reduction
        # )
        # self.norm_logits = tf.Variable(tf.zeros([512, 93431]), dtype="float32", trainable=False)
        # self.y_true = tf.Variable(tf.zeros([512, 93431], dtype="int32"), dtype="int32", trainable=False)

    def build(self, batch_size):
        self.batch_labels_back_up = tf.Variable(tf.zeros([batch_size], dtype="int64"), dtype="int64", trainable=False)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        # norm_logits = y_pred
        # self.norm_logits.assign(norm_logits)
        # self.y_true.assign(y_true)
        # pick_cond = tf.cast(y_true, dtype=tf.bool)
        # y_pred_vals = norm_logits[pick_cond]
        pick_cond = tf.where(y_true != 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        # tf.print(", y_true.sum:", tf.reduce_sum(y_true), ", y_pred_vals.shape:", K.shape(y_pred_vals), ", y_true.shape:", K.shape(y_true), end="")
        # tf.assert_equal(tf.reduce_sum(y_true), K.shape(y_true)[0])
        # tf.assert_equal(K.shape(y_pred_vals)[0], K.shape(y_true)[0])
        # y_pred_vals = tf.clip_by_value(y_pred_vals, -1, 1)
        if self.margin1 == 1.0 and self.margin2 == 0.0 and self.margin3 == 0.0:
            theta = y_pred_vals
        elif self.margin1 == 1.0 and self.margin3 == 0.0:
            theta = tf.cos(tf.acos(y_pred_vals) + self.margin2)
        else:
            theta = tf.cos(tf.acos(y_pred_vals) * self.margin1 + self.margin2) - self.margin3
            # Grad(theta) == 0
            #   ==> np.sin(np.math.acos(xx) * margin1 + margin2) == 0
            #   ==> np.math.acos(xx) * margin1 + margin2 == np.pi
            #   ==> xx == np.cos((np.pi - margin2) / margin1)
            #   ==> theta_min == theta(xx) == -1 - margin3
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)
        # theta_one_hot = tf.expand_dims(theta_valid - y_pred_vals, 1) * tf.cast(y_true, dtype=tf.float32)
        # arcface_logits = (theta_one_hot + norm_logits) * self.scale
        # theta_one_hot = tf.expand_dims(theta_valid, 1) * tf.cast(y_true, dtype=tf.float32)
        # arcface_logits = tf.where(pick_cond, theta_one_hot, norm_logits) * self.scale
        arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale
        # tf.assert_equal(tf.math.is_nan(tf.reduce_mean(arcface_logits)), False)
        # arcface_logits = tf.cond(tf.math.is_finite(tf.reduce_mean(arcface_logits)), lambda: arcface_logits, lambda: tf.cast(y_true, "float32"))
        # arcface_logits = tf.where(tf.math.is_finite(arcface_logits), arcface_logits, tf.zeros_like(arcface_logits))
        # cond = tf.repeat(tf.math.is_finite(tf.reduce_sum(arcface_logits, axis=-1, keepdims=True)), arcface_logits.shape[-1], axis=-1)
        # arcface_logits = tf.where(cond, arcface_logits, tf.zeros_like(arcface_logits))
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)
        # return self.reduction_func(y_true, arcface_logits)

    def get_config(self):
        config = super(ArcfaceLoss, self).get_config()
        config.update(
            {
                "margin1": self.margin1,
                "margin2": self.margin2,
                "margin3": self.margin3,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ArcfaceLoss simple
@keras.utils.register_keras_serializable(package="keras_insightface")
class ArcfaceLossSimple(tf.keras.losses.Loss):
    def __init__(self, margin=0.5, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(ArcfaceLossSimple, self).__init__(**kwargs)
        self.margin, self.scale, self.from_logits, self.label_smoothing = margin, scale, from_logits, label_smoothing
        self.margin_cos, self.margin_sin = tf.cos(margin), tf.sin(margin)
        self.threshold = tf.cos(np.pi - margin)
        # self.low_pred_punish = tf.sin(np.pi - margin) * margin
        self.theta_min = -2
        self.batch_labels_back_up = None

    def build(self, batch_size):
        self.batch_labels_back_up = tf.Variable(tf.zeros([batch_size], dtype="int64"), dtype="int64", trainable=False)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        pick_cond = tf.where(y_true != 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        theta = y_pred_vals * self.margin_cos - tf.sqrt(1 - tf.pow(y_pred_vals, 2)) * self.margin_sin
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)
        arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super(ArcfaceLossSimple, self).get_config()
        config.update(
            {
                "margin": self.margin,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# [CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition](https://arxiv.org/pdf/2004.00288.pdf)
@keras.utils.register_keras_serializable(package="keras_insightface")
class CurricularFaceLoss(ArcfaceLossSimple):
    def __init__(self, margin=0.5, scale=64.0, from_logits=True, label_smoothing=0, hard_scale=0, **kwargs):
        super(CurricularFaceLoss, self).__init__(margin, scale, from_logits, label_smoothing, **kwargs)
        self.hard_scale = tf.Variable(hard_scale, dtype="float32", trainable=False)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        pick_cond = tf.where(y_true != 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        theta = y_pred_vals * self.margin_cos - tf.sqrt(1 - tf.pow(y_pred_vals, 2)) * self.margin_sin
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)

        self.hard_scale.assign(tf.reduce_mean(y_pred_vals) * 0.01 + (1 - 0.01) * self.hard_scale)
        tf.print(", hard_scale:", self.hard_scale, end="")
        hard_norm_logits = tf.where(norm_logits > tf.expand_dims(theta_valid, 1), norm_logits * (self.hard_scale + norm_logits), norm_logits)

        arcface_logits = tf.tensor_scatter_nd_update(hard_norm_logits, pick_cond, theta_valid) * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super(CurricularFaceLoss, self).get_config()
        config.update({"hard_scale": K.get_value(self.hard_scale)})
        return config


# [AirFace:Lightweight and Efficient Model for Face Recognition](https://arxiv.org/pdf/1907.12256.pdf)
@keras.utils.register_keras_serializable(package="keras_insightface")
class AirFaceLoss(ArcfaceLossSimple):
    def __init__(self, margin=0.4, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(AirFaceLoss, self).__init__(margin, scale, from_logits, label_smoothing, **kwargs)
        # theta = (np.pi - 2 * (tf.acos(y_pred_vals).numpy() + margin)) / np.pi
        #   ==> theta = 1 - tf.acos(y_pred_vals).numpy() * 2 / np.pi - 2 * margin / np.pi
        #   ==> theta = 1 - 2 * margin / np.pi - tf.acos(y_pred_vals).numpy() * 2 / np.pi
        self.margin_head = 1 - 2 * margin / np.pi
        self.margin_scale = 2 / np.pi

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        # norm_logits = tf.acos(norm_logits) * self.margin_scale
        # logits = tf.where(tf.cast(y_true, dtype=tf.bool), self.margin_head - norm_logits, 1 - norm_logits) * self.scale
        pick_cond = tf.where(y_true != 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        theta = self.margin_head - tf.acos(y_pred_vals) * self.margin_scale
        logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta) * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)


# [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf)
@keras.utils.register_keras_serializable(package="keras_insightface")
class CosFaceLoss(ArcfaceLossSimple):
    def __init__(self, margin=0.35, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(CosFaceLoss, self).__init__(margin, scale, from_logits, label_smoothing, **kwargs)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        pick_cond = tf.cast(y_true, dtype=tf.bool)
        logits = tf.where(pick_cond, norm_logits - self.margin, norm_logits) * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)


# [MagFace: A Universal Representation for Face Recognition and Quality Assessment](https://arxiv.org/pdf/2103.06627.pdf)
@keras.utils.register_keras_serializable(package="keras_insightface")
class MagFaceLoss(ArcfaceLossSimple):
    """Another set for fine-tune is: min_feature_norm, max_feature_norm, min_margin, max_margin, regularizer_loss_lambda = 1, 51, 0.45, 1, 5"""

    def __init__(
        self,
        min_feature_norm=10.0,
        max_feature_norm=110.0,
        min_margin=0.45,
        max_margin=0.8,
        scale=64.0,
        regularizer_loss_lambda=35.0,
        use_cosface_margin=False,
        curricular_hard_scale=-1,
        from_logits=True,
        label_smoothing=0,
        **kwargs
    ):
        super(MagFaceLoss, self).__init__(scale=scale, from_logits=from_logits, label_smoothing=label_smoothing, **kwargs)
        # l_a, u_a, lambda_g
        self.min_feature_norm, self.max_feature_norm, self.regularizer_loss_lambda = min_feature_norm, max_feature_norm, regularizer_loss_lambda
        # l_margin, u_margin
        self.min_margin, self.max_margin = min_margin, max_margin
        self.use_cosface_margin, self.curricular_hard_scale = use_cosface_margin, curricular_hard_scale
        self.margin_scale = (max_margin - min_margin) / (max_feature_norm - min_feature_norm)
        self.regularizer_loss_scale = 1.0 / (self.max_feature_norm**2)
        self.use_curricular_scale = False
        self.epislon = 1e-3
        if curricular_hard_scale >= 0:
            self.curricular_hard_scale = tf.Variable(curricular_hard_scale, dtype="float32", trainable=False)
            self.use_curricular_scale = True
        # np.set_printoptions(precision=4)
        # self.precission_4 = lambda xx: tf.math.round(xx * 10000) / 10000

    def call(self, y_true, norm_logits_with_norm):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        # feature_norm is multiplied with -1 in NormDense layer, keeping low for not affecting accuracy metrics.
        norm_logits, feature_norm = norm_logits_with_norm[:, :-1], norm_logits_with_norm[:, -1] * -1
        norm_logits = tf.clip_by_value(norm_logits, -1 + self.epislon, 1 - self.epislon)
        feature_norm = tf.clip_by_value(feature_norm, self.min_feature_norm, self.max_feature_norm)
        # margin = (self.u_margin-self.l_margin) / (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        margin = self.margin_scale * (feature_norm - self.min_feature_norm) + self.min_margin
        margin = tf.expand_dims(margin, 1)

        pick_cond = tf.where(y_true != 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        if self.use_cosface_margin:
            # Cosface process
            arcface_logits = tf.where(tf.cast(y_true, dtype=tf.bool), norm_logits - margin, norm_logits) * self.scale
            # theta_valid = y_pred_vals - margin
        else:
            # Arcface process
            margin_cos, margin_sin = tf.cos(margin), tf.sin(margin)
            # XLA after TF > 2.7.0 not supporting this gather_nd -> tensor_scatter_nd_update method...
            # threshold = tf.cos(np.pi - margin)
            # theta = y_pred_vals * margin_cos - tf.sqrt(tf.maximum(1 - tf.pow(y_pred_vals, 2), 0.0)) * margin_sin
            # theta_valid = tf.where(y_pred_vals > threshold, theta, self.theta_min - theta)
            # arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale
            arcface_logits = tf.where(
                tf.cast(y_true, dtype=tf.bool),
                norm_logits * margin_cos - tf.sqrt(tf.maximum(1 - tf.pow(norm_logits, 2), 0.0)) * margin_sin,
                norm_logits,
            )
            arcface_logits = tf.minimum(arcface_logits, norm_logits) * self.scale

        # if self.use_curricular_scale:
        #     self.curricular_hard_scale.assign(tf.reduce_mean(y_pred_vals) * 0.01 + (1 - 0.01) * self.curricular_hard_scale)
        #     tf.print(", hard_scale:", self.curricular_hard_scale, end="")
        #     norm_logits = tf.where(norm_logits > tf.expand_dims(theta_valid, 1), norm_logits * (self.curricular_hard_scale + norm_logits), norm_logits)

        arcface_loss = tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

        # MegFace loss_G, g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        regularizer_loss = self.regularizer_loss_scale * feature_norm + 1.0 / feature_norm

        tf.print(
            # ", regularizer_loss: ",
            # tf.reduce_mean(regularizer_loss),
            ", arcface: ",
            tf.reduce_mean(arcface_loss),
            ", margin: ",
            tf.reduce_mean(margin),
            # ", min: ",
            # tf.reduce_min(margin),
            # ", max: ",
            # tf.reduce_max(margin),
            ", feature_norm: ",
            tf.reduce_mean(feature_norm),
            # ", min: ",
            # tf.reduce_min(feature_norm),
            # ", max: ",
            # tf.reduce_max(feature_norm),
            sep="",
            end="\r",
        )
        return arcface_loss + regularizer_loss * self.regularizer_loss_lambda

    def get_config(self):
        config = super(MagFaceLoss, self).get_config()
        config.update(
            {
                "min_feature_norm": self.min_feature_norm,
                "max_feature_norm": self.max_feature_norm,
                "min_margin": self.min_margin,
                "max_margin": self.max_margin,
                "regularizer_loss_lambda": self.regularizer_loss_lambda,
                "use_cosface_margin": self.use_cosface_margin,
                "curricular_hard_scale": K.get_value(self.curricular_hard_scale),
            }
        )
        return config


# [AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/pdf/2204.00964.pdf)
@keras.utils.register_keras_serializable(package="keras_insightface")
class AdaFaceLoss(ArcfaceLossSimple):
    """
    margin_alpha:
      - When margin_alpha=0.33, the model performs the best. For 0.22 or 0.66, the performance is still higher.
      - As long as h is set such that ∥dzi∥ has some variation, margin_alpha is not very sensitive.

    margin:
      - The performance is best for HQ datasets when margin=0.4, for LQ datasets when margin=0.75.
      - Large margin results in large angular margin variation based on the image quality, resulting in more adaptivity.

    mean_std_alpha: Update pace for batch_mean and batch_std.
    """

    def __init__(self, margin=0.4, margin_alpha=0.333, mean_std_alpha=0.01, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super().__init__(scale=scale, from_logits=from_logits, label_smoothing=label_smoothing, **kwargs)
        self.min_feature_norm, self.max_feature_norm, self.epislon = 0.001, 100, 1e-3
        self.margin, self.margin_alpha, self.mean_std_alpha = margin, margin_alpha, mean_std_alpha
        self.batch_mean = tf.Variable(20, dtype="float32", trainable=False)
        self.batch_std = tf.Variable(100, dtype="float32", trainable=False)
        self.cos_max_epislon = tf.acos(-1.0) - self.epislon  # pi - epislon

    def __to_scaled_margin__(self, feature_norm):
        norm_mean = tf.math.reduce_mean(feature_norm)
        samples = tf.cast(tf.maximum(1, feature_norm.shape[0] - 1), feature_norm.dtype)
        norm_std = tf.sqrt(tf.math.reduce_sum((feature_norm - norm_mean) ** 2) / samples)  # Torch std
        self.batch_mean.assign(self.mean_std_alpha * norm_mean + (1.0 - self.mean_std_alpha) * self.batch_mean)
        self.batch_std.assign(self.mean_std_alpha * norm_std + (1.0 - self.mean_std_alpha) * self.batch_std)
        margin_scaler = (feature_norm - self.batch_mean) / (self.batch_std + self.epislon)  # 66% between -1, 1
        margin_scaler = tf.clip_by_value(margin_scaler * self.margin_alpha, -1, 1)  # 68% between -0.333 ,0.333 when h:0.333
        return tf.expand_dims(self.margin * margin_scaler, 1)

    def call(self, y_true, norm_logits_with_norm):
        if self.batch_labels_back_up is not None:  # For VPL mode
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        # feature_norm is multiplied with -1 in NormDense layer, keeping low for not affecting accuracy metrics.
        norm_logits, feature_norm = norm_logits_with_norm[:, :-1], norm_logits_with_norm[:, -1] * -1
        norm_logits = tf.clip_by_value(norm_logits, -1 + self.epislon, 1 - self.epislon)
        feature_norm = tf.clip_by_value(feature_norm, self.min_feature_norm, self.max_feature_norm)
        scaled_margin = tf.stop_gradient(self.__to_scaled_margin__(feature_norm))
        # tf.print(", margin: ", tf.reduce_mean(scaled_margin), sep="", end="\r")
        tf.print(", margin hist: ", tf.histogram_fixed_width(scaled_margin, [-self.margin, self.margin], nbins=3), sep="", end="\r")
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # XLA after TF > 2.7.0 not supporting gather_nd -> tensor_scatter_nd_update method...
        arcface_logits = tf.where(
            tf.cast(y_true, dtype=tf.bool),
            tf.cos(tf.clip_by_value(tf.acos(norm_logits) - scaled_margin, self.epislon, self.cos_max_epislon)) - (self.margin + scaled_margin),
            norm_logits,
        )
        # arcface_logits = tf.minimum(arcface_logits, norm_logits) * self.scale
        arcface_logits *= self.scale

        # return arcface_logits
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "margin": self.margin,
                "margin_alpha": self.margin_alpha,
                "mean_std_alpha": self.mean_std_alpha,
                "_batch_mean_": float(self.batch_mean.numpy()),
                "_batch_std_": float(self.batch_std.numpy()),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        _batch_mean_ = config.pop("_batch_mean_", 20.0)
        _batch_std_ = config.pop("_batch_std_", 100.0)
        aa = cls(**config)
        aa.batch_mean.assign(_batch_mean_)
        aa.batch_std.assign(_batch_std_)
        return aa


# [AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations](https://arxiv.org/pdf/1905.00292.pdf)
@keras.utils.register_keras_serializable(package="keras_insightface")
class AdaCosLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=1000, scale=0, max_median=np.pi / 4, from_logits=True, label_smoothing=0, **kwargs):
        super(AdaCosLoss, self).__init__(**kwargs)
        self.max_median, self.from_logits, self.label_smoothing = max_median, from_logits, label_smoothing
        self.num_classes = num_classes
        self.theta_med_max = tf.cast(max_median, "float32")
        if scale == 0:
            scale = tf.sqrt(2.0) * tf.math.log(float(num_classes) - 1)
        self.scale = tf.Variable(scale, dtype="float32", trainable=False)

    @tf.function
    def call(self, y_true, norm_logits):
        pick_cond = tf.cast(y_true, dtype=tf.bool)
        y_pred_vals = norm_logits[pick_cond]
        theta = tf.acos(y_pred_vals)
        med_pos = tf.shape(norm_logits)[0] // 2 - 1
        theta_med = tf.sort(theta)[med_pos]

        B_avg = tf.where(pick_cond, tf.zeros_like(norm_logits), tf.exp(self.scale * norm_logits))
        B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1))
        self.scale.assign(tf.math.log(B_avg) / tf.cos(tf.minimum(self.theta_med_max, theta_med)))
        tf.print(", scale:", self.scale, "theta_med:", theta_med, end="")

        arcface_logits = norm_logits * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)
        # return self.reduction_func(y_true, arcface_logits)

    def get_config(self):
        config = super(AdaCosLoss, self).get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "scale": K.get_value(self.scale),
                "max_median": self.max_median,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Combination of Arcface loss and Triplet loss
@keras.utils.register_keras_serializable(package="keras_insightface")
class AcrTripLoss(ArcfaceLossSimple):
    def __init__(self, margin=0.5, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(AcrTripLoss, self).__init__(margin, scale, from_logits, label_smoothing, **kwargs)

    def call(self, y_true, outputs):
        # tf.print(">>>> ", outputs.shape)
        embeddings = outputs[:, :512]
        norm_logits = outputs[:, 512:]
        """ Triplet part """
        labels = tf.argmax(y_true, axis=1)
        pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        norm_emb = tf.nn.l2_normalize(embeddings, 1)
        dists = tf.matmul(norm_emb, tf.transpose(norm_emb))
        pos_dists = tf.where(pos_mask, dists, tf.ones_like(dists))
        pos_hardest_dists = tf.reduce_min(pos_dists, -1)

        pos_hd_margin = tf.cos(tf.acos(pos_hardest_dists) + self.margin)
        pos_hd_margin_valid = tf.where(pos_hardest_dists > self.threshold, pos_hd_margin, self.theta_min - pos_hd_margin)

        neg_dists = tf.where(pos_mask, tf.ones_like(dists) * -1, dists)
        neg_hardest_dists = tf.reduce_max(neg_dists, -1)
        triplet_loss = tf.maximum(neg_hardest_dists - pos_hardest_dists, 0.0)

        """ Arcface part """
        pick_cond = tf.where(y_true != 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        theta = tf.cos(tf.acos(y_pred_vals) + self.margin)

        """ Combine """
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta) - triplet_loss
        arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)


# Callback to save center values on each epoch end
class Save_Numpy_Callback(tf.keras.callbacks.Callback):
    def __init__(self, save_file, save_tensor):
        super(Save_Numpy_Callback, self).__init__()
        self.save_file = os.path.splitext(save_file)[0]
        self.save_tensor = save_tensor

    def on_epoch_end(self, epoch=0, logs=None):
        np.save(self.save_file, self.save_tensor.numpy())


# [A Discriminative Feature Learning Approach for Deep Face Recognition](http://ydwen.github.io/papers/WenECCV16.pdf)
@keras.utils.register_keras_serializable(package="keras_insightface")
class CenterLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, emb_shape=512, alpha=0.5, initial_file=None, **kwargs):
        super(CenterLoss, self).__init__(**kwargs)
        self.num_classes, self.emb_shape, self.alpha = num_classes, emb_shape, alpha
        self.initial_file = initial_file
        centers = tf.Variable(tf.zeros([num_classes, emb_shape]), trainable=False, aggregation=tf.VariableAggregation.MEAN)
        # centers = tf.Variable(tf.random.truncated_normal((num_classes, emb_shape)), trainable=False, aggregation=tf.VariableAggregation.MEAN)
        if initial_file:
            if os.path.exists(initial_file):
                print(">>>> Reload from center backup:", initial_file)
                aa = np.load(initial_file)
                centers.assign(aa)
            self.save_centers_callback = Save_Numpy_Callback(initial_file, centers)
        self.centers = centers
        if tf.distribute.has_strategy():
            self.num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        else:
            self.num_replicas = 1

    def __calculate_center_loss__(self, centers_batch, embedding):
        return tf.reduce_sum(tf.square(embedding - centers_batch), axis=-1) / 2

    def call(self, y_true, embedding):
        # embedding = y_pred[:, : self.emb_shape]
        labels = tf.argmax(y_true, axis=1)
        centers_batch = tf.gather(self.centers, labels)
        # loss = tf.reduce_mean(tf.square(embedding - centers_batch))
        loss = self.__calculate_center_loss__(centers_batch, embedding)

        # Update centers
        diff = centers_batch - embedding
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.cast(tf.gather(unique_count, unique_idx), tf.float32)

        # diff = diff / tf.expand_dims(appear_times, 1)
        diff = diff / tf.expand_dims(appear_times + 1, 1)  # Δcj
        diff = self.num_replicas * self.alpha * diff
        # print(centers_batch.shape, self.centers.shape, labels.shape, diff.shape)
        self.centers.assign(tf.tensor_scatter_nd_sub(self.centers, tf.expand_dims(labels, 1), diff))
        # centers_batch = tf.gather(self.centers, labels)
        return loss

    def get_config(self):
        config = super(CenterLoss, self).get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "emb_shape": self.emb_shape,
                "alpha": self.alpha,
                "initial_file": self.initial_file,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "feature_dim" in config:
            config["emb_shape"] = config.pop("feature_dim")
        if "factor" in config:
            config.pop("factor")
        if "logits_loss" in config:
            config.pop("logits_loss")
        return cls(**config)


@keras.utils.register_keras_serializable(package="keras_insightface")
class CenterLossCosine(CenterLoss):
    def __calculate_center_loss__(self, centers_batch, embedding):
        norm_emb = tf.nn.l2_normalize(embedding, 1)
        norm_center = tf.nn.l2_normalize(centers_batch, 1)
        return 1 - tf.reduce_sum(norm_emb * norm_center, axis=-1)


# TripletLoss helper class definitions [Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)
@keras.utils.register_keras_serializable(package="keras_insightface")
class TripletLossWapper(tf.keras.losses.Loss):
    def __init__(self, alpha=0.35, **kwargs):
        # reduction = tf.keras.losses.Reduction.NONE if tf.distribute.has_strategy() else tf.keras.losses.Reduction.AUTO
        # super(TripletLossWapper, self).__init__(**kwargs, reduction=reduction)
        super(TripletLossWapper, self).__init__(**kwargs)
        self.alpha = alpha

    def __calculate_triplet_loss__(self, y_true, y_pred, alpha):
        return None

    def call(self, labels, embeddings):
        return self.__calculate_triplet_loss__(labels, embeddings, self.alpha)

    def get_config(self):
        config = super(TripletLossWapper, self).get_config()
        config.update({"alpha": self.alpha})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package="keras_insightface")
class BatchHardTripletLoss(TripletLossWapper):
    def __calculate_triplet_loss__(self, labels, embeddings, alpha):
        labels = tf.argmax(labels, axis=1)
        # labels = tf.squeeze(labels)
        # labels.set_shape([None])
        pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        norm_emb = tf.nn.l2_normalize(embeddings, 1)
        dists = tf.matmul(norm_emb, tf.transpose(norm_emb))
        # pos_dists = tf.ragged.boolean_mask(dists, pos_mask)
        pos_dists = tf.where(pos_mask, dists, tf.ones_like(dists))
        pos_hardest_dists = tf.reduce_min(pos_dists, -1)
        # neg_dists = tf.ragged.boolean_mask(dists, tf.logical_not(pos_mask))
        neg_dists = tf.where(pos_mask, tf.ones_like(dists) * -1, dists)
        neg_hardest_dists = tf.reduce_max(neg_dists, -1)
        basic_loss = neg_hardest_dists - pos_hardest_dists + alpha
        # ==> pos - neg > alpha
        # ==> neg + alpha - pos < 0
        # return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return tf.maximum(basic_loss, 0.0)


# Triplet loss using arcface margin
@keras.utils.register_keras_serializable(package="keras_insightface")
class ArcBatchHardTripletLoss(TripletLossWapper):
    def __init__(self, alpha=0.35, **kwargs):
        super(ArcBatchHardTripletLoss, self).__init__(alpha=alpha, **kwargs)
        # self.margin = alpha
        self.threshold = tf.cos(np.pi - alpha)
        self.theta_min = -2

    def __calculate_triplet_loss__(self, labels, embeddings, alpha):
        labels = tf.argmax(labels, axis=1)
        # labels = tf.squeeze(labels)
        # labels.set_shape([None])
        pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        norm_emb = tf.nn.l2_normalize(embeddings, 1)
        dists = tf.matmul(norm_emb, tf.transpose(norm_emb))
        # pos_dists = tf.ragged.boolean_mask(dists, pos_mask)
        pos_dists = tf.where(pos_mask, dists, tf.ones_like(dists))
        pos_hardest_dists = tf.reduce_min(pos_dists, -1)

        pos_hd_margin = tf.cos(tf.acos(pos_hardest_dists) + self.alpha)
        pos_hd_margin_valid = tf.where(pos_hardest_dists > self.threshold, pos_hd_margin, self.theta_min - pos_hd_margin)

        # neg_dists = tf.ragged.boolean_mask(dists, tf.logical_not(pos_mask))
        neg_dists = tf.where(pos_mask, tf.ones_like(dists) * -1, dists)
        neg_hardest_dists = tf.reduce_max(neg_dists, -1)
        tf.print(
            " - triplet_dists_mean:",
            tf.reduce_mean(dists),
            "pos:",
            tf.reduce_mean(pos_hardest_dists),
            "pos_hd_valid:",
            tf.reduce_mean(pos_hd_margin_valid),
            "neg:",
            tf.reduce_mean(neg_hardest_dists),
            end="\r",
        )
        # basic_loss = neg_hardest_dists - pos_hd_margin_valid + alpha
        basic_loss = neg_hardest_dists - pos_hd_margin_valid
        return tf.maximum(basic_loss, 0.0)


@keras.utils.register_keras_serializable(package="keras_insightface")
class BatchHardTripletLossEuclidean(TripletLossWapper):
    def __calculate_triplet_loss__(self, labels, embeddings, alpha):
        labels = tf.argmax(labels, axis=1)
        pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        # dense_mse_func = lambda xx: tf.reduce_sum(tf.square((embeddings - xx)), axis=-1)
        # dense_mse_func = tf.function(dense_mse_func, input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
        # dists = tf.vectorized_map(dense_mse_func, embeddings)

        # Euclidean_dists = aa ** 2 + bb ** 2 - 2 * aa * bb, where aa = embeddings, bb = embeddings
        embeddings_sqaure_sum = tf.reduce_sum(tf.square(embeddings), axis=-1)
        ab = tf.matmul(embeddings, tf.transpose(embeddings))
        dists = tf.reshape(embeddings_sqaure_sum, (-1, 1)) + embeddings_sqaure_sum - 2 * ab
        # pos_dists = tf.ragged.boolean_mask(dists, pos_mask)
        pos_dists = tf.where(pos_mask, dists, tf.zeros_like(dists))
        pos_hardest_dists = tf.reduce_max(pos_dists, -1)
        # neg_dists = tf.ragged.boolean_mask(dists, tf.logical_not(pos_mask))
        neg_dists = tf.where(pos_mask, tf.ones_like(dists) * tf.reduce_max(dists), dists)
        neg_hardest_dists = tf.reduce_min(neg_dists, -1)
        tf.print(
            " - triplet_dists_mean:",
            tf.reduce_mean(dists),
            "pos:",
            tf.reduce_mean(pos_hardest_dists),
            "neg:",
            tf.reduce_mean(neg_hardest_dists),
            end="",
        )
        basic_loss = pos_hardest_dists + alpha - neg_hardest_dists
        # ==> neg - pos > alpha
        # ==> pos + alpha - neg < 0
        # return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return tf.maximum(basic_loss, 0.0)


@keras.utils.register_keras_serializable(package="keras_insightface")
class BatchHardTripletLossEuclideanAutoAlpha(TripletLossWapper):
    def __init__(self, alpha=0.1, init_auto_alpha=1, **kwargs):
        # reduction = tf.keras.losses.Reduction.NONE if tf.distribute.has_strategy() else tf.keras.losses.Reduction.AUTO
        # super(TripletLossWapper, self).__init__(**kwargs, reduction=reduction)
        super(BatchHardTripletLossMSEAutoAlpha, self).__init__(alpha=alpha, **kwargs)
        self.auto_alpha = tf.Variable(init_auto_alpha, dtype="float", trainable=False)

    def __calculate_triplet_loss__(self, labels, embeddings, alpha):
        labels = tf.argmax(labels, axis=1)
        pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        # dense_mse_func = lambda xx: tf.reduce_sum(tf.square((embeddings - xx)), axis=-1)
        # dense_mse_func = tf.function(dense_mse_func, input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
        # dists = tf.vectorized_map(dense_mse_func, embeddings)

        # Euclidean_dists = aa ** 2 + bb ** 2 - 2 * aa * bb, where aa = embeddings, bb = embeddings
        embeddings_sqaure_sum = tf.reduce_sum(tf.square(embeddings), axis=-1)
        ab = tf.matmul(embeddings, tf.transpose(embeddings))
        dists = tf.reshape(embeddings_sqaure_sum, (-1, 1)) + embeddings_sqaure_sum - 2 * ab
        # pos_dists = tf.ragged.boolean_mask(dists, pos_mask)
        pos_dists = tf.where(pos_mask, dists, tf.zeros_like(dists))
        pos_hardest_dists = tf.reduce_max(pos_dists, -1)
        # neg_dists = tf.ragged.boolean_mask(dists, tf.logical_not(pos_mask))
        neg_dists = tf.where(pos_mask, tf.ones_like(dists) * tf.reduce_max(dists), dists)
        neg_hardest_dists = tf.reduce_min(neg_dists, -1)
        basic_loss = pos_hardest_dists + self.auto_alpha - neg_hardest_dists
        self.auto_alpha.assign(tf.reduce_mean(dists) * alpha)
        tf.print(
            " - triplet_dists_mean:",
            tf.reduce_mean(dists),
            "pos:",
            tf.reduce_mean(pos_hardest_dists),
            "neg:",
            tf.reduce_mean(neg_hardest_dists),
            "auto_alpha:",
            self.auto_alpha,
            end="",
        )
        # ==> neg - pos > alpha
        # ==> pos + alpha - neg < 0
        # return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return tf.maximum(basic_loss, 0.0)


@keras.utils.register_keras_serializable(package="keras_insightface")
class BatchAllTripletLoss(TripletLossWapper):
    def __calculate_triplet_loss__(self, labels, embeddings, alpha):
        labels = tf.argmax(labels, axis=1)
        # labels = tf.squeeze(labels)
        # labels.set_shape([None])
        pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        norm_emb = tf.nn.l2_normalize(embeddings, 1)
        dists = tf.matmul(norm_emb, tf.transpose(norm_emb))

        pos_dists = tf.where(pos_mask, dists, tf.ones_like(dists))
        pos_dists_loss = tf.reduce_sum(1.0 - pos_dists, -1) / tf.reduce_sum(tf.cast(pos_mask, dtype=tf.float32), -1)
        pos_hardest_dists = tf.expand_dims(tf.reduce_min(pos_dists, -1), 1)

        neg_valid_mask = tf.logical_and(tf.logical_not(pos_mask), (pos_hardest_dists - dists) < alpha)
        neg_dists_valid = tf.where(neg_valid_mask, dists, tf.zeros_like(dists))
        neg_dists_loss = tf.reduce_sum(neg_dists_valid, -1) / (tf.reduce_sum(tf.cast(neg_valid_mask, dtype=tf.float32), -1) + 1)
        return pos_dists_loss + neg_dists_loss


@keras.utils.register_keras_serializable(package="keras_insightface")
class OfflineTripletLoss(TripletLossWapper):
    def __calculate_triplet_loss__(self, labels, embeddings, alpha):
        norm_emb = tf.nn.l2_normalize(embeddings, 1)
        # [anchor, pos, neg, anchor, pos, neg] -> [pos, pos], [neg, neg]]
        anchor_emb, pos_emb, neg_emb = tf.split(tf.reshape(norm_emb, [-1, 3, norm_emb.shape[-1]]), 3, axis=1)
        anchor_emb, pos_emb, neg_emb = anchor_emb[:, 0], pos_emb[:, 0], neg_emb[:, 0]
        # anchor_emb, pos_emb, neg_emb = norm_emb[::3], norm_emb[1::3], norm_emb[2::3]

        pos_dist = tf.reduce_sum(anchor_emb * pos_emb, -1)
        neg_dist = tf.reduce_sum(anchor_emb * neg_emb, -1)
        basic_loss = neg_dist - pos_dist + alpha
        return tf.maximum(basic_loss, 0.0)


@keras.utils.register_keras_serializable(package="keras_insightface")
class OfflineArcTripletLoss(TripletLossWapper):
    def __init__(self, alpha=0.35, **kwargs):
        super().__init__(alpha=alpha, **kwargs)
        self.threshold = tf.cos(np.pi - alpha)
        self.theta_min = -2

    def __calculate_triplet_loss__(self, labels, embeddings, alpha):
        norm_emb = tf.nn.l2_normalize(embeddings, 1)
        # [anchor, pos, neg, anchor, pos, neg] -> [pos, pos], [neg, neg]]
        anchor_emb, pos_emb, neg_emb = norm_emb[::3], norm_emb[1::3], norm_emb[2::3]

        pos_dist = tf.reduce_sum(tf.multiply(anchor_emb, pos_emb), -1)
        neg_dist = tf.reduce_sum(tf.multiply(anchor_emb, neg_emb), -1)

        pos_margin = tf.cos(tf.acos(pos_dist) + self.alpha)
        pos_valid = tf.where(pos_margin > self.threshold, pos_margin, self.theta_min - pos_margin)

        basic_loss = neg_dist - pos_valid
        return tf.maximum(basic_loss, 0.0)


@keras.utils.register_keras_serializable(package="keras_insightface")
def distiller_loss_euclidean(true_emb, pred_emb):
    return tf.reduce_sum(tf.square(pred_emb - true_emb), axis=-1)


@keras.utils.register_keras_serializable(package="keras_insightface")
def distiller_loss_cosine(true_emb, pred_emb):
    # tf.print(true_emb.shape, true_emb.dtype, pred_emb.shape, pred_emb.dtype)
    norm_one = tf.sqrt(1.0 / tf.cast(true_emb.shape[-1], true_emb.dtype))
    true_emb = tf.where(tf.math.is_finite(true_emb), true_emb, tf.zeros_like(true_emb) + norm_one)
    true_norm_value = tf.norm(true_emb, axis=-1) + 1e-5
    pred_norm_value = tf.norm(pred_emb, axis=-1) + 1e-5
    true_emb_normed = true_emb / tf.expand_dims(true_norm_value, -1)
    pred_emb_normed = pred_emb / tf.expand_dims(pred_norm_value, -1)
    # tf.assert_equal(tf.math.is_nan(tf.reduce_mean(true_emb)), False, message='nan in true_emb_normed')
    cosine_loss = 1 - tf.reduce_sum(pred_emb_normed * true_emb_normed, axis=-1)
    return cosine_loss
    # l2_loss = tf.abs(true_norm_value - pred_norm_value)
    # tf.print("cos_loss:", cosine_loss, "l2_loss:", l2_loss)
    # return cosine_loss + l2_loss * 0.001
    # return tf.reduce_sum(tf.square(pred_emb - true_emb), axis=-1)


# [PDF 2106.05237 Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/pdf/2106.05237.pdf)
@keras.utils.register_keras_serializable(package="keras_insightface")
class DistillKLDivergenceLoss(tf.keras.losses.Loss):
    def __init__(self, scale=10, **kwargs):
        super(DistillKLDivergenceLoss, self).__init__(**kwargs)
        self.scale = scale
        self.kl_divergence = tf.keras.losses.KLDivergence()

    def call(self, teacher_prob, student_prob):
        # return self.kl_divergence(teacher_prob * self.scale, student_prob * self.scale)
        # return self.kl_divergence(
        #     tf.nn.softmax(teacher_prob / self.temperature, axis=1),
        #     tf.nn.softmax(student_prob / self.temperature, axis=1),
        # )

        # teacher_prob for NormDense layer value [-0.5, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # For scale = 0.1, softmax result [0.1310, 0.1377, 0.1405, 0.1433, 0.1462, 0.1491, 0.1521]
        # For scale = 1, softmax result [0.0547, 0.0902, 0.1102, 0.1346, 0.1643, 0.2007, 0.2452]
        # For scale = 10, softmax result [2.6450-07, 3.9256-05, 0.0003, 0.0021, 0.0158, 0.1170, 0.8646]
        # For scale = 20, softmax result [9.1862-14, 2.0234-09, 1.1047-07, 6.0317-06, 0.0003, 0.0179, 0.9816]
        return self.kl_divergence(
            tf.nn.softmax(teacher_prob * self.scale, axis=1),
            tf.nn.softmax(student_prob * self.scale, axis=1),
        )
