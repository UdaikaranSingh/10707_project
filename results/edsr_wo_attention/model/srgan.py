from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from model.attention import Conv2D_attention, ChannelAttention, SpatialAttention

from model.common import pixel_shuffle, normalize_01, normalize_m11, denormalize_m11

LR_SIZE = 24
HR_SIZE = 96


def upsample(x_in, num_filters, attention=False):
    if not attention:
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    else:
        # x = Conv2D_attention(x_in, numFilters=num_filters, kernelSize=3)
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = ChannelAttention(num_filters, 3)(x)
        x = SpatialAttention(8)(x)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)


def res_block(x_in, num_filters, momentum=0.8, attention=False):
    if not attention:
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    else:
        # x = Conv2D_attention(x_in, numFilters=num_filters, kernelSize=3)
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = ChannelAttention(num_filters, 3)(x)
        x = SpatialAttention(8)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    
    if not attention:
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    else:
        # x = Conv2D_attention(x, numFilters=num_filters, kernelSize=3)
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = ChannelAttention(num_filters, 3)(x)
        x = SpatialAttention(8)(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x


def sr_resnet(num_filters=64, num_res_blocks=16, attention=False):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize_01)(x_in)

    if not attention:
        x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    else:
        # x = Conv2D_attention(x, numFilters=num_filters, kernelSize=9)
        x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
        x = ChannelAttention(num_filters, 9)(x)
        x = SpatialAttention(8)(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters, attention=attention)

    if not attention:
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    else:
        # x = Conv2D_attention(x, numFilters=num_filters, kernelSize=3)
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = ChannelAttention(num_filters, 3)(x)
        x = SpatialAttention(8)(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample(x, num_filters * 4, attention=attention)
    x = upsample(x, num_filters * 4, attention=attention)

    if not attention:
        x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    else:
        # x = Conv2D_attention(x, numFilters=num_filters, kernelSize=3, activation='tanh')
        x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
        x = ChannelAttention(3, 9)(x)
        x = SpatialAttention(8)(x)
    x = Lambda(denormalize_m11)(x)

    return Model(x_in, x)


generator = sr_resnet


def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8, attention=False):
    if not attention:
        x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    else:
        # x = Conv2D_attention(x_in, numFilters=num_filters, kernelSize=3, strides=strides)
        x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
        x = ChannelAttention(num_filters, 3)(x)
        x = SpatialAttention(8)(x)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)


def discriminator(num_filters=64, attention=False):
    x_in = Input(shape=(HR_SIZE, HR_SIZE, 3))
    x = Lambda(normalize_m11)(x_in)

    x = discriminator_block(x, num_filters, batchnorm=False, attention=attention)
    x = discriminator_block(x, num_filters, strides=2, attention=attention)

    x = discriminator_block(x, num_filters * 2, attention=attention)
    x = discriminator_block(x, num_filters * 2, strides=2, attention=attention)

    x = discriminator_block(x, num_filters * 4, attention=attention)
    x = discriminator_block(x, num_filters * 4, strides=2, attention=attention)

    x = discriminator_block(x, num_filters * 8, attention=attention)
    x = discriminator_block(x, num_filters * 8, strides=2, attention=attention)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)


def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)
