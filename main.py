import tensorflow as tf
import keras
import keras.backend as KK
from keras.models import Model
from keras import layers
from keras.datasets import mnist


# TODO review paper to determine how the two types of blocks stack

alpha = 1.0
expansion_rate = 6
output_channels_k = 3
batch_size = 96
weight_decay = 0.00004
learning_rate = 0.045
dropout_rate = 1e-3
num_classes = 10 #1000


img_rows, img_cols, img_channels = 28, 28, 1 #224, 244, 3

full_model = True

ds_name_template = 'ds_{}'
bn_name_template = 'bn_{}'


# TODO batch normalization and dropout
def downsample(block_id, inputs, filters):
    # Downsample block
    ds_name = ds_name_template.format(block_id) + '_'
    hidden = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, name=ds_name + 'conv0')(inputs)
    hidden = layers.BatchNormalization(name=ds_name + 'bn0')(hidden)
    hidden = layers.ReLU(6., name=ds_name + 'relu0')(hidden)
    hidden = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=2, name=ds_name + 'conv1')(hidden)
    hidden = layers.BatchNormalization(name=ds_name + 'bn1')(hidden)
    hidden = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, name=ds_name + 'conv2')(hidden)
    hidden = layers.BatchNormalization(name=ds_name + 'bn2')(hidden)

    return hidden


def bottleneck(block_id, inputs, output_filters, expansion_factor):
    # Inverted residual block
    bn_name = bn_name_template.format(block_id) + '_'
    input_shape = KK.eval(KK.shape(inputs)[-1])
    expanded_filters = output_filters*expansion_factor
    hidden = layers.Conv2D(filters=expanded_filters, kernel_size=(1, 1), strides=1, padding='same', name=bn_name + "conv0")(inputs)
    hidden = layers.BatchNormalization(name=bn_name + 'bn0')(hidden)
    hidden = layers.ReLU(6., name=bn_name + 'relu0')(hidden)
    hidden = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', name=bn_name + 'depth_conv0')(hidden)
    hidden = layers.BatchNormalization(name=bn_name + 'bn1')(hidden)
    hidden = layers.ReLU(6., name=bn_name + 'relu1')(hidden)
    hidden = layers.Conv2D(filters=output_filters, kernel_size=(1, 1), strides=1, padding='same', name=bn_name + 'conv1')(hidden)
    hidden = layers.BatchNormalization(name=bn_name + 'bn2')(hidden)

    if input_shape == output_filters:
        print("************** Residual connection made. **************")
        hidden = layers.Add(name=bn_name + 'add0')([inputs, hidden])
    else:
        print("!!!!!!!!!!!!!! No residual connection made. !!!!!!!!!!!!!!")
    print("input_shape: {}\texpanded_filters: {}".format(input_shape, expanded_filters))

    return hidden


input_layer = layers.Input(batch_shape=(batch_size, img_rows, img_cols, img_channels), name='input')

if full_model:
    padded_input = layers.ZeroPadding2D(padding=((0, 196), (0, 196)))(input_layer)
else:
    padded_input = input_layer

# TODO pad bottom and left of input using zero padding
conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, name="input_conv")(padded_input)
batch_norm1 = layers.BatchNormalization(name="batch_norm")(conv1)

# Block 0 - Input: 112^2 x 32
bn0_1 = bottleneck(0, batch_norm1, 16, 1)

# Block 1
ds1 = downsample(1, bn0_1, 24)
bn1_2 = bottleneck(1, ds1, 24, 6)

# Block 2
ds2 = downsample(2, bn1_2, 32)
bn2_1 = bottleneck('2a', ds2, 32, 6)
bn2_2 = bottleneck('2b', bn2_1, 32, 6)

if full_model:
    # Block 3
    ds3 = downsample(3, bn2_2, 64)
    bn3_1 = bottleneck('3a', ds3, 64, 6)
    bn3_2 = bottleneck('3b', bn3_1, 64, 6)
    bn3_3 = bottleneck('3c', bn3_2, 64, 6)

    # Block 4
    bn4_1 = bottleneck(4, bn3_3, 96, 6)
    bn4_2 = bottleneck('4a', bn4_1, 96, 6)
    bn4_3 = bottleneck('4b', bn4_2, 96, 6)

    # Block 5
    ds5 = downsample(5, bn4_3, 160)
    bn5_1 = bottleneck('5a', ds5, 160, 6)
    bn5_2 = bottleneck('5b', bn5_1, 160, 6)

    # Block 6
    bn6_1 = bottleneck('6a', bn5_2, 320, 6)
    bn6_2 = bottleneck('6b', bn6_1, 320, 6)

    conv2 = layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=1, name="final_conv")(bn6_2)
    batch_norm2 = layers.BatchNormalization()(conv2)

    #avg_pool1 = layers.AveragePooling2D(pool_size=(7, 7), strides=1)(conv2)
    avg_pool1 = layers.GlobalAveragePooling2D()(batch_norm2)


    dense1 = layers.Dense(num_classes, use_bias=True, name='Logits')(avg_pool1)
else:
    conv2 = layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=1, name="final_conv")(bn2_2)
    batch_norm2 = layers.BatchNormalization()(conv2)

    avg_pool1 = layers.GlobalAveragePooling2D()(batch_norm2)
    dense1 = layers.Dense(num_classes, use_bias=True, name='Logits')(avg_pool1)

output = layers.Softmax(name="output")(dense1)

model = Model([input_layer], [output])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

'''
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


frozen_graph = freeze_session(KK.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "/tmp/mobilenetv2", "my_model.pb", as_text=False)
'''

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
input_shape = (img_rows, img_cols, img_channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model.fit(x_train, y_train, batch_size=batch_size, epochs=3, verbose=1, validation_data=(x_test, y_test))