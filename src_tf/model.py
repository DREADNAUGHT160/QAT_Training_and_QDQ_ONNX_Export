import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, models, Input

def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """
    A standard ResNet block.
    """
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x

def resnet_stack(x, filters, blocks, stride1=2, name=None):
    """
    A stack of ResNet blocks.
    """
    x = resnet_block(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = resnet_block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x

def ResNet18(input_shape=(32, 32, 3), num_classes=10):
    """
    ResNet18 architecture for CIFAR-10.
    """
    inputs = Input(shape=input_shape)
    
    # Initial Conv
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(axis=3, name='bn_conv1')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)

    # ResNet Stages
    x = resnet_stack(x, 64, 2, stride1=1, name='conv2')
    x = resnet_stack(x, 128, 2, name='conv3')
    x = resnet_stack(x, 256, 2, name='conv4')
    x = resnet_stack(x, 512, 2, name='conv5')

    # Output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='fc1000')(x)

    model = models.Model(inputs, outputs, name='resnet18')
    return model
