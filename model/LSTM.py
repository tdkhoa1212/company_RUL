from tensorflow.keras.layers import Conv1D, Activation, \
                                    Dense, BatchNormalization,\
                                    Input, MaxPooling1D, Lambda, \
                                    MaxPooling2D, Flatten, Dropout, LSTM, AveragePooling1D
import tensorflow as tf
from keras.models import Model
from keras import layers, regularizers
import keras.backend as K
from tensorflow_addons.layers import MultiHeadAttention


def TransformerLayer(x, c, num_heads=16, training=None):
    x = tf.keras.layers.Dense(c,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(x)
    x = Dropout(0.5)(x, training=training)
    ma  = MultiHeadAttention(head_size=num_heads, num_heads=num_heads)([x, x, x]) 
    ma = tf.keras.layers.Dense(c,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(ma) 
    ma = Dropout(0.5)(ma, training=training)
    ma = tf.keras.layers.Dense(c,  activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(ma) 
    ma = Dropout(0.5)(ma, training=training)
    ma = tf.keras.layers.Bidirectional(LSTM(units=c, return_sequences=False, activation='relu', recurrent_dropout=0.2))(ma)
    ma = Dropout(0.5)(ma, training=training)
    return ma

def identity_block(input_tensor, kernel_size, filters, stage, block, training):
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5),
              name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=training)
    x = Activation('relu')(x)

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5),
              name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=training)

    if input_tensor.shape[2] != x.shape[2]:
        x = layers.add([x, Lambda(lambda y: K.repeat_elements(y, rep=2, axis=2))(input_tensor)])
    else:
        x = layers.add([x, input_tensor])

    x = BatchNormalization()(x, training=training)
    x = Activation('relu')(x)
    return x

def lstm_model(opt, training=None, inputs=None):
  x = Conv1D(64,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
  x = BatchNormalization()(x, training=training)
  x = Activation('relu')(x)
  x = AveragePooling1D(pool_size=4, strides=None, padding='valid')(x)
  
  for i in range(3):
    x = identity_block(x, kernel_size=3, filters=64, stage=1, block=i, training=training)

  x = AveragePooling1D(pool_size=4, strides=None, padding='valid')(x)

  for i in range(4):
    x = identity_block(x, kernel_size=3, filters=128, stage=2, block=i, training=training)

  x = AveragePooling1D(pool_size=4, strides=None, padding='valid')(x)

  for i in range(23):
    x = identity_block(x, kernel_size=3, filters=256, stage=3, block=i, training=training)

  x = MaxPooling1D(pool_size=4, strides=None, padding='valid')(x)

  for i in range(3):
    x = identity_block(x, kernel_size=3, filters=512, stage=4, block=i, training=training)
    
  return x

def lstm_extracted_model(opt, training=None, inputs=None):
  x = Conv1D(28,
               kernel_size=4,
               strides=1,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
  x = BatchNormalization()(x, training=training)
  x = Activation('relu')(x)
  x = AveragePooling1D(pool_size=2, strides=None, padding='valid')(x)

  x = Conv1D(56,
               kernel_size=2,
               strides=1,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001))(x)
  x = BatchNormalization()(x, training=training)
  x = Activation('relu')(x)
  x = AveragePooling1D(pool_size=2, strides=None, padding='valid')(x)
  return x
