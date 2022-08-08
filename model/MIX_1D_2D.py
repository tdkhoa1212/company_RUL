from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM
import tensorflow as tf
from keras.models import Model
from tensorflow_addons.layers import MultiHeadAttention
from keras import layers, regularizers
import keras.backend as K

def TransformerLayer(q, v, k, num_heads=4, training=None):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    q = tf.keras.layers.Dense(256,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(q)
    k = tf.keras.layers.Dense(256,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(k)
    v = tf.keras.layers.Dense(256,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(v)
    ma  = MultiHeadAttention(head_size=num_heads, num_heads=num_heads)([q, k, v])
    ma = BatchNormalization()(ma, training=training)
    ma = Activation('relu')(ma) 
    return ma

def add(x1, x2, x3, training=None):
    x1 = tf.keras.layers.Dense(1024,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(x1)
    x2 = tf.keras.layers.Dense(1024,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(x2)
    x3 = tf.keras.layers.Dense(1024,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(x3)
    x = x1 + x2 + x3  
    return x

def mix_model(opt, cnn_1d_model, resnet_50, lstm_extracted_model, input_1D, input_2D, input_extracted, training=False):
  out_1D = Dropout(0.1)(cnn_1d_model(opt, training, input_1D), training=training)
  out_2D = Dropout(0.1)(resnet_50(opt)(input_2D, training=training), training=training)
  out_extracted = lstm_extracted_model(opt, training, input_extracted)
  
  network_1D = Model(input_1D, out_1D, name='network_1D')
  network_2D = Model(input_2D, out_2D, name='network_2D')
  network_extracted = Model(input_extracted, out_extracted, name='network_extracted')
  
  hidden_out_1D = network_1D([input_1D])
  hidden_out_2D = network_2D([input_2D])
  hidden_out_extracted = network_extracted([input_extracted])
  
  merged_value_0 = TransformerLayer(hidden_out_1D, hidden_out_2D, hidden_out_extracted, 16, training)
  merged_value_1 = TransformerLayer(hidden_out_1D, hidden_out_2D, hidden_out_extracted, 16, training)
    
  Condition = Dense(3, 
                    activation='softmax', 
                    name='Condition')(merged_value_0)
  RUL = Dense(1, 
              activation='sigmoid', 
              name='RUL')(merged_value_0)
  return Condition, RUL
