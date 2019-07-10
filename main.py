import keras
import keras.backend as KK
from keras.models import Model
from keras import layers


# TODO review paper to determine how the two types of blocks stack

expansion_rate = 6

relu6 = layers.activations.relu(max_value=6)


# TODO batch normalization and dropout
def downsample(inputs, filters, expansion_factor):
    # Downsample block
    hidden = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1)(hidden)
    hidden = relu6(hidden)
    hidden = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=2)(hidden)
    hidden = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1)(hidden)

    return hidden

def bottleneck(inputs, output_filters, expansion_factor):
    # Inverted residual block
    input_shape = KK.shape(inputs)[-1]
    expanded_filters = input_shape*expansion_factor
    hidden = layers.Conv2D(filters=expanded_filters, kernel_size=(1, 1), strides=1)(inputs)
    hidden = relu6(hidden)
    hidden = layers.SeparableConv2D(filters=expanded_filters, kernel_size=(3, 3), strides=1)(hidden)
    hidden = relu6(hidden)
    hidden = layers.Conv2D(filters=output_filters, kernel_size=(1, 1), strides=1)(hidden)
    hidden = layers.Add()([inputs, hidden])

    return hidden

input_layer = layers.Input((224, 224, 3), name='input')
conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2)(input_layer)

# Block 0 - Input: 112^2 x 32
bn0_1 = bottleneck(conv1, 16, 1)

# Block 1
ds1 = downsample(bn0_1, 24, 6)
bn1_2 = bottleneck(ds1, 24, 6)

# Block 2
ds2 = downsample(bn1_2, 32, 6)
bn2_1 = bottleneck(ds2, 32, 6)
bn2_2 = bottleneck(bn2_1, 32, 6)

# Block 3
ds3 = downsample(bn2_2, 64, 6)
bn3_1 = bottleneck(ds3, 64, 6)
bn3_2 = bottleneck(bn3_1, 64, 6)
bn3_3 = bottleneck(bn3_1, 64, 6)

# Block 4
bn4_1 = bottleneck(bn3_3, 96, 6)
bn4_2 = bottleneck(bn4_1, 96, 6)
bn4_3 = bottleneck(bn4_2, 96, 6)

# Block 5
ds5 = downsample(bn4_3, 160, 6)
bn5_1 = bottleneck(ds5, 160, 6)
bn5_2 = bottleneck(bn5_2, 160, 6)

# Block 6
ds6 = downsample(bn5_2, 320, 6)
bn6_1 = bottleneck(ds6, 320, 6)
bn6_2 = bottleneck(bn6_1, 320, 6)
