from model.MIX_1D_2D import mix_model
from model.resnet import resnet_101
from model.LSTM import lstm_extracted_model, lstm_model
from utils.tools import to_onehot
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
import tensorflow as tf
import argparse
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--save_dir',       default='/content/drive/MyDrive/Khoa/vibration_project/RUL/results', type=str)
    parser.add_argument('--data_type',      default=['2d', '1d', 'extract'], type=list, help='shape of data. They can be 1d, 2d, extract')
    parser.add_argument('--train_bearing',  default=['Bearing1_1', 'Bearing1_2', 'Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2'], type=str, nargs='+')   
    parser.add_argument('--test_bearing',   default=['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5'], type=str, nargs='+')
    parser.add_argument('--scaler',         default=None, type=str)
    parser.add_argument('--main_dir_colab', default='/content/drive/MyDrive/Khoa/data/', type=str)

    parser.add_argument('--epochs',         default=30, type=int)
    parser.add_argument('--batch_size',     default=16, type=int)
    parser.add_argument('--input_shape',    default=32768, type=int, help='1279 for using fft, 2560 for raw data in PHM, 32768 for raw data in XJTU')
    
    parser.add_argument('--load_weight',  default=False, type=bool)  
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

# Train and test for PHM data ############################################################################################
def main(opt, train_1D, train_2D, train_extract, train_label_RUL, test_1D, test_2D, test_extract, test_label_RUL):  
  input_extracted = Input((14, 2), name='Extracted_LSTM_input')
  input_1D = Input((opt.input_shape, 2), name='LSTM_CNN1D_input')
  input_2D = Input((128, 128, 2), name='CNN_input')
  Condition, RUL = mix_model_PHM(opt, lstm_model, resnet_101, lstm_extracted_model, input_1D, input_2D, input_extracted, True)
  network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=[Condition, RUL])

  # get three types of different forms from original data-------------------------------
  val_2D, val_1D, val_extract, val_label_RUL = test_2D[:300], test_1D[:300], test_extract[:300], test_label_RUL[:300]
  val_data =   [val_1D, val_2D, val_extract]
  train_data = [train_1D, train_2D, train_extract]
  test_data  = [test_1D, test_2D, test_extract]
  
  weight_path = os.path.join(opt.save_dir, f'model_{opt.condition}_{opt.type}')
  if opt.load_weight:
    if os.path.exists(weight_path):
      print(f'\nLoad weight: {weight_path}\n')
      network.load_weights(weight_path)
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=1)
  network.compile(optimizer=tf.keras.optimizers.RMSprop(1e-4),
                  loss=tf.keras.losses.MeanSquaredLogarithmicError(), 
                  metrics=['mae', tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()], 
                  # run_eagerly=True
                    ) # https://keras.io/api/losses/ 
  network.summary()

  # dataset_train = tf.data.Dataset.from_tensor_slices((train_data , train_label)).batch(opt.batch_size)
  tf.debugging.set_log_device_placement(True)

  history = network.fit(train_data , train_label_RUL,
                        epochs     = opt.epochs,
                        batch_size = opt.batch_size,
                        validation_data = (val_data, val_label_RUL))
  network.save(weight_path)
  _, RUL_mae, RUL_r_square, RUL_mean_squared_error = network.evaluate(test_data, test_label_RUL, verbose=0)
  RUL_mae = round(RUL_mae, 4)
  RUL_r_square = round(RUL_r_square, 4)
  RUL_mean_squared_error = round(RUL_mean_squared_error, 4)
  print(f'\n----------Score in test set: \n mae: {RUL_mae}, r2: {RUL_r_square}, rmse: {RUL_mean_squared_error}\n' )

if __name__ == '__main__':
  opt = parse_opt()
  from utils.load_XJTU_data import train_1D, train_2D, train_extract, train_label_RUL,\
                                    test_1D, test_2D, test_extract, test_label_RUL
  main(opt, train_1D, train_2D, train_extract, train_label_RUL, test_1D, test_2D, test_extract, test_label_RUL)
