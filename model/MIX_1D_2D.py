from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM
import tensorflow as tf
from keras.models import Model
from keras import layers, regularizers
import keras.backend as K

from tensorflow_addons.layers import MultiHeadAttention
# def TransformerLayer(x, num_heads=4, training=None):
#     x = tf.keras.layers.Dense(56,   activation='relu',
#                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                                      bias_regularizer=regularizers.l2(1e-4),
#                                      activity_regularizer=regularizers.l2(1e-5))(x)
#     x = Dropout(0.1)(x, training=training)
#     ma  = MultiHeadAttention(head_size=num_heads, num_heads=num_heads)([x, x, x]) 
#     ma = BatchNormalization()(ma, training=training)
#     ma = tf.keras.layers.Dense(56,   activation='relu',
#                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                                      bias_regularizer=regularizers.l2(1e-4),
#                                      activity_regularizer=regularizers.l2(1e-5))(ma) 
#     ma = tf.keras.layers.Dense(56,  activation='relu',
#                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                                      bias_regularizer=regularizers.l2(1e-4),
#                                      activity_regularizer=regularizers.l2(1e-5))(ma) 
#     ma = Dropout(0.1)(ma, training=training)
#     ma = tf.keras.layers.GRU(56, return_sequences=False)(ma)
#     ma = Dropout(0.1)(ma, training=training)
#     return ma

def fully_concatenate(hidden_out_1D, hidden_out_2D, hidden_out_extracted, training):
    hidden_out_1D = tf.keras.layers.Dense(1024,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(hidden_out_1D) 
    hidden_out_2D = tf.keras.layers.Dense(512,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(hidden_out_2D) 
    hidden_out_extracted = tf.keras.layers.Dense(64,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(hidden_out_extracted) 
    hidden_out_1D = Dropout(0.1)(hidden_out_1D, training=training)
    hidden_out_2D = Dropout(0.1)(hidden_out_2D, training=training)
    hidden_out_extracted = Dropout(0.1)(hidden_out_extracted, training=training)
    
    all_ = concatenate((hidden_out_1D, hidden_out_2D, hidden_out_extracted))
#     all_ = tf.keras.layers.Dense(1024,   activation='relu',
#                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                                      bias_regularizer=regularizers.l2(1e-4),
#                                      activity_regularizer=regularizers.l2(1e-5))(all_) 
#     all_ = Dropout(0.1)(all_, training=training)
    return all_

def mix_model(opt, cnn_1d_model, resnet_50, lstm_extracted_model, input_1D, input_2D, input_extracted, training=False):
  out_1D = cnn_1d_model(opt, training, input_1D)
  out_2D = resnet_50(opt)(input_2D, training=training)
  out_extracted = lstm_extracted_model(opt, training, input_extracted)
  
  network_1D = Model(input_1D, out_1D, name='network_1D')
  network_2D = Model(input_2D, out_2D, name='network_2D')
  network_extracted = Model(input_extracted, out_extracted, name='network_extracted')
  
  hidden_out_1D = network_1D([input_1D])
  hidden_out_2D = network_2D([input_2D])
  hidden_out_extracted = network_extracted([input_extracted])
  
  merged_value_1 = fully_concatenate(hidden_out_1D, hidden_out_2D, hidden_out_extracted, training)
#   merged_value_0 = fully_concatenate(hidden_out_1D, hidden_out_2D, hidden_out_extracted, training)
  # merged_value_2 = TransformerLayer(hidden_out_1D, hidden_out_2D, hidden_out_extracted, 8, training)
  # merged_value_3 = TransformerLayer(hidden_out_1D, hidden_out_2D, hidden_out_extracted, 8, training)
  # merged_value_4 = concatenate((merged_value_1, merged_value_2, merged_value_3))
#   hidden_out_extracted = tf.keras.layers.Dense(256,  activation='relu',
#                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                                      bias_regularizer=regularizers.l2(1e-4),
#                                      activity_regularizer=regularizers.l2(1e-5))(hidden_out_extracted) 
#   merged_value_1 = Dropout(0.1)(hidden_out_extracted, training=training)
    
  Condition = Dense(3, 
                    activation='softmax', 
                    name='Condition')(merged_value_1)
  RUL = Dense(1, 
              activation='sigmoid', 
              name='RUL')(merged_value_1)
  return Condition, RUL
