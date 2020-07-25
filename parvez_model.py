from functools import partial

from keras.layers import Input, ELU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.engine import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss


create_convolution_block = partial(create_convolution_block, activation=ELU, instance_normalization=True)


def parvez_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=4, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=3, optimizer=Adam, initial_learning_rate=3e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    """
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)
    dilation_rate=(2, 2, 2)
    #dilation_rate1=(3, 3, 3)
   # dilation_rate2=(5, 5, 5)
    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dilation_rate, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    def parvez1(level_number):
        input_layer=level_output_layers[level_number]
        convolution1 = create_convolution_block(input_layer=input_layer, n_filters=level_filters[level_number])
        dropout = SpatialDropout3D(rate=dropout_rate, data_format="channels_first")(convolution1)
        convolution2 = create_convolution_block(input_layer=dropout, n_filters=12, dilation_rate=dilation_rate)
        convolution2 = concatenate([dropout, convolution2], axis=1)
        convolution3 = create_convolution_block(input_layer=dropout, n_filters=12, dilation_rate=(3, 3, 3))
        convolution3 = concatenate([dropout, convolution3], axis=1)
        convolution4 = create_convolution_block(input_layer=dropout, n_filters=12, dilation_rate=(5, 5, 5))
        convolution4 = concatenate([dropout, convolution4], axis=1)
        concat = concatenate([convolution2, convolution3, convolution4], axis=1)
        return concat
        
    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        #if(level_number==3):
         #    a=parvez1(level_number)
          #   print("X", a)
           #  b=create_convolution_block(a, level_filters[level_number], kernel=(1, 1, 1))
            # print("Y", b)
        if(level_number==2):
             a=parvez1(level_number)
          #   print("Z", a)
             b=create_convolution_block(a, level_filters[level_number], kernel=(1, 1, 1))
             print("XX", b)
        elif(level_number==1):
             a=parvez1(level_number)
           #  print("XY", a)
             b=create_convolution_block(a, level_filters[level_number], kernel=(1, 1, 1))
             print("YZ", b)
        elif(level_number==0):
             a=parvez1(level_number)
            # print("ZX", a)
             b=create_convolution_block(a, level_filters[level_number], kernel=(1, 1, 1))
             print("ZY", b)
        concatenation_layer = concatenate([b, up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
           if(level_number==0):
              segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        if(level_number==0):
              segmentation_layer = segmentation_layers[level_number]
              if output_layer is None:
                 output_layer = segmentation_layer

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    print(model.summary())
   # plot_model(model, to_file='./model.png')
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
  #  dropout = SpatialDropout3D(rate=dropout_rate, data_format="channels_first")(convolution1)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters,  size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters, kernel=(1, 1, 1))
    return convolution


def create_context_module(input_layer, n_level_filters, dilation_rate, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters, dilation_rate=dilation_rate)
  #  convolution3 = create_convolution_block(input_layer=convolution2, n_filters=n_level_filters)
    return convolution2
