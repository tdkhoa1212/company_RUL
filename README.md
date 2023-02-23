# RUL_predictor

## 1. Instroduction
## 2. Dataset and result
## 3. Guide to run code
### 3.1. Install prerequisite
> Download git repository

    git clone https://github.com/tdkhoa1212/company_RUL.git

>Enter the git folder

    cd company_RUL

>Install libraries
    pip install -r requirements.txt
### 3.2. Training process
#### Parameters: 
**--save_dir**,     &emsp;  default='/content/drive/MyDrive/Khoa/vibration_project/RUL/results'<br/>
**--data_type**,    &emsp;  default=['2d', '1d', 'extract']<br/>
**--train_bearing**,&emsp;  default=['Bearing1_1', 'Bearing1_2', 'Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']<br/>
**--test_bearing**, &emsp;  default=['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5']<br/>
**--scaler**,       &emsp;  default='Normalizer'<br/>
**--main_dir_colab**,&emsp; default='/content/drive/MyDrive/Khoa/data/'<br/>
**--epochs**,       &emsp;  default=30<br/>
**--batch_size**,   &emsp;  default=16<br/>
**--input_shape**,  &emsp;  default=32768<br/>
**--load_weight**,  &emsp;  default=False<br/>
### 3.3. Testing process
## 4. Citation: TODO
