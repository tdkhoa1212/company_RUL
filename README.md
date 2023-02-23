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
'--save_dir',       default='/content/drive/MyDrive/Khoa/vibration_project/RUL/results'
'--data_type',      default=['2d', '1d', 'extract']
'--train_bearing',  default=['Bearing1_1', 'Bearing1_2', 'Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
'--test_bearing',   default=['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5']
--scaler',         default='Normalizer'
### 3.3. Testing process
## 4. Citation: TODO
