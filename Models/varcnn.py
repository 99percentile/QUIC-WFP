# Code for standard ResNet model is based on
# https://github.com/broadinstitute/keras-resnet
def ResNet18(inputs, suffix, blocks=None, block=None, numerical_names=None):
    """Constructs a `keras.models.Model` object using the given block count.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param block: a residual block (e.g. an instance of
        `keras_resnet.blocks.basic_2d`)
    :param numerical_names: list of bool, same size as blocks, used to
        indicate whether names of layers should include numbers or letters
    :return model: ResNet model with encoding output (if `include_top=False`)
        or classification output (if `include_top=True`)
    """
    if blocks is None:
        blocks = [2, 2, 2, 2]
    if block is None:
        block = dilated_basic_1d
    if numerical_names is None:
        numerical_names = [True] * len(blocks)
    
    x = ZeroPadding1D(padding=3, name='padding_conv1_' + suffix)(inputs)
    x = Conv1D(64, 7, strides=2, use_bias=False, name='conv1_' + suffix)(x)
    x = BatchNormalization(epsilon=1e-5, name='bn_conv1_' + suffix)(x)
    x = Activation('relu', name='conv1_relu_' + suffix)(x)
    x = MaxPooling1D(3, strides=2, padding='same', name='pool1_' + suffix)(x)
    
    features = 64
    outputs = []
    
    for stage_id, iterations in enumerate(blocks):
        x = block(features, suffix, stage_id, 0, dilations=(1, 2),
                  numerical_name=False)(x)
        for block_id in range(1, iterations):
            x = block(features, suffix, stage_id, block_id, dilations=(4, 8),
                      numerical_name=(
                              block_id > 0 and numerical_names[stage_id]))(
                x)
        
        features *= 2
        outputs.append(x)
    
    x = GlobalAveragePooling1D(name='pool5_' + suffix)(x)
    return x

use_dir = useDirection
use_time = useTime
use_size = useLength
use_tcp = useTcp
use_quic = useQuic
use_burst = useTotalBurst

seq_length = seq_len

dir_dilations = True
time_dilations = True
size_dilations = True
tcp_dilations = True
quic_dilations = True
burst_dilations = True

num_mon_sites = num_domains
num_unmon_sites = 0

base_patience = 5

if use_dir:
    dir_input = Input(shape=(seq_length, 1,), name='dir_input')
    if dir_dilations:
        dir_output = ResNet18(dir_input, 'dir', block=dilated_basic_1d)
    else:
        dir_output = ResNet18(dir_input, 'dir', block=basic_1d)

# Constructs time ResNet
if use_time:
    time_input = Input(shape=(seq_length, 1,), name='time_input')
    if time_dilations:
        time_output = ResNet18(time_input, 'time', block=dilated_basic_1d)
    else:
        time_output = ResNet18(time_input, 'time', block=basic_1d)

# Constructs size ResNet
if use_size:
    size_input = Input(shape=(seq_length, 1,), name='size_input')
    if size_dilations:
        size_output = ResNet18(size_input, 'size', block=dilated_basic_1d)
    else:
        size_output = ResNet18(size_input, 'size', block=basic_1d)

# Constructs tcp ResNet
if use_tcp:
    tcp_input = Input(shape=(seq_length, 1,), name='tcp_input')
    if tcp_dilations:
        tcp_output = ResNet18(tcp_input, 'tcp', block=dilated_basic_1d)
    else:
        tcp_output = ResNet18(tcp_input, 'tcp', block=basic_1d)

# Constructs quic ResNet
if use_quic:
    quic_input = Input(shape=(seq_length, 1,), name='quic_input')
    if quic_dilations:
        quic_output = ResNet18(quic_input, 'quic', block=dilated_basic_1d)
    else:
        quic_output = ResNet18(quic_input, 'quic', block=basic_1d)

# Constructs  ResNet
if use_burst:
    burst_input = Input(shape=(seq_length, 1,), name='burst_input')
    if burst_dilations:
        burst_output = ResNet18(burst_input, 'burst', block=dilated_basic_1d)
    else:
        burst_output = ResNet18(burst_input, 'burst', block=basic_1d)

# Forms input and output lists and possibly add final dense layer
input_params = []
concat_params = []

if use_size:
    input_params.append(size_input)
    concat_params.append(size_output)

if use_time:
    input_params.append(time_input)
    concat_params.append(time_output)

if use_dir:
    input_params.append(dir_input)
    concat_params.append(dir_output)

if use_tcp:
    input_params.append(tcp_input)
    concat_params.append(tcp_output)

if use_quic:
    input_params.append(quic_input)
    concat_params.append(quic_output)

if use_burst:
    input_params.append(burst_input)
    concat_params.append(burst_output)


if len(concat_params) == 1:
    combined = concat_params[0]
else:
    combined = Concatenate()(concat_params)

# Better to have final fc layer if combining multiple models
if len(concat_params) > 1:
    combined = Dense(1024)(combined)
    combined = BatchNormalization()(combined)
    combined = Activation('relu')(combined)
    combined = Dropout(0.5)(combined)

output_classes = num_mon_sites if num_unmon_sites == 0 else num_mon_sites + 1
model_output = Dense(units=output_classes, activation='softmax',
                        name='model_output')(combined)