import sys
sys.path.append('../')
from utils.lib import *

def create_model(maxv):
    inputs = Input(shape=(48,800,1))
 
    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)

    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    pool_6 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_6)

    conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
    batch_norm_6 = BatchNormalization()(conv_7)
    pool_7 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
    conv_8 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_7)
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_8)

    blstm_1 = LSTM(512, return_sequences=True, dropout = 0.3)(squeezed)
    blstm_2 = LSTM(512, return_sequences=True, dropout = 0.3)(blstm_1)

    outputs = Dense(96, activation = 'softmax')(blstm_2)

    act_model = Model(inputs, outputs)
    
#     labels = Input(name='the_labels', shape=[maxv], dtype='float32')
#     input_length = Input(name='input_length', shape=[1], dtype='int64')
#     label_length = Input(name='label_length', shape=[1], dtype='int64')
#     loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
#     model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)

#     adam = Adam(learning_rate=0.0002,clipvalue=0.5,clipnorm=1.0)

#     model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = adam)
    
    return act_model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
 
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def decode(output,chars):
    decoded_txt = []
    for item in output:
        curr_list = ''
        for i in item:
            if i != -1:
                curr_list+=chars[int(i)]
        decoded_txt.append(str(curr_list))
    return decoded_txt