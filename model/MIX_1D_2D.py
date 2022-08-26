from tensorflow.keras.layers import Conv1D, Activation, Dense, \
                                    concatenate, BatchNormalization, \
                                    GlobalAveragePooling1D, Input, \
                                    MaxPooling1D, Lambda, GlobalAveragePooling2D, \
                                    ReLU, MaxPooling2D, Flatten, Dropout, LSTM, Reshape
import tensorflow as tf
from keras.models import Model
from model.LSTM import TransformerLayer
from keras import layers, regularizers
import keras.backend as K

def fully_concatenate(hidden_out_1D, hidden_out_2D, hidden_out_extracted):
    all_ = concatenate((hidden_out_1D, hidden_out_2D, hidden_out_extracted))
    return all_

def reshape(x):
    out = Reshape((x.shape[-2]*x.shape[-3], x.shape[-1]))(x)
    return out

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
  
  rul_hidden_out_1D = TransformerLayer(hidden_out_1D, hidden_out_1D.shape[-1], training=training)
  rul_hidden_out_2D = reshape(hidden_out_2D)
  rul_hidden_out_2D = TransformerLayer(rul_hidden_out_2D, rul_hidden_out_2D.shape[-1], training=training)
  rul_hidden_out_extracted = TransformerLayer(hidden_out_extracted, hidden_out_extracted.shape[-1], training=training)
  
  con_hidden_out_1D = GlobalAveragePooling1D()(hidden_out_1D)
  con_hidden_out_2D = GlobalAveragePooling2D()(hidden_out_2D)
  con_hidden_out_extracted = GlobalAveragePooling1D()(hidden_out_extracted)
  
  merged_value_0 = fully_concatenate(rul_hidden_out_1D, rul_hidden_out_2D, rul_hidden_out_extracted)
  merged_value_1 = fully_concatenate(con_hidden_out_1D, con_hidden_out_2D, con_hidden_out_extracted)
    
  Condition = Dense(3, 
                    activation='softmax', 
                    name='Condition')(merged_value_1)
  RUL = Dense(1, 
              activation='sigmoid', 
              name='RUL')(merged_value_0)
  return Condition, RUL
