# RUL_predictor
## Main steps to integrate this tool to the system:
1. Doing 3.1 section
2. Doing 3.3 section

## 1. Architecture
![plot](./images/architecture.png)
## 2. Data Construction
- data
    - data001(machine001/Fan_NDE)
        - long(machine001/Fan_NDE/long_interval_data)
            - 1.csv
            - 2.csv
            - ...
        - short.csv(machine001/Fan_NDE/short_interval_data/short_int....csv)
    - data002(machine001/Motor_DE/long_interval_data)
    - data003(machine002/Motor_DE/long_interval_data)
    - data004(machine003/Fan_NDE/long_interval_data)
    - data005(machine004/Fan_NDE/long_interval_data)</br>

***NOTE: *** 
- label of data in long interval data folder is made based on short interval data
- data is used for training and testing according to 5 folds strategy with 4 for training and 1 for testing
## 3. Guide to run code
### 3.1. Install prerequisite
> Download git repository

    git clone https://github.com/tdkhoa1212/company_RUL.git

>Enter the git folder

    cd company_RUL

>Install libraries
    pip install -r requirements.txt

</br>

### 3.2. Training process
#### Parameters in the train.py file: 
**--save_dir**,     &emsp;  default='./results', help='the directory to save results'<br/>
**--data_type**,    &emsp;  default=['2d', '1d', 'extract'], help='types of input data'<br/>
**--train_bearing**,&emsp;  default=['Bearing1_1', 'Bearing1_2', 'Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2'], help='data used for traing'<br/>
**--test_bearing**, &emsp;  default=['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5'], help='data used for testing'<br/>
**--scaler**,       &emsp;  default='Normalizer', help='a type of normalizing method'<br/>
**--main_dir_colab**,&emsp; default='./data' , help='the directory to the database'<br/>
**--epochs**,       &emsp;  default=50, help='the number of iteration'<br/>
**--batch_size**,   &emsp;  default=16, help='the batch of data for training process'<br/>
**--input_shape**,  &emsp;  default=32768, help='the length of a typical segment of vibration signal'<br/>
**--load_weight**,  &emsp;  default=False, help='set true if you want to use the exist weight file'<br/>

***__Note__:*** there are some parameters needing to be changed including ***save_dir, train_bearing, test_bearing, main_dir_colab, input_shape, load_weight***
#### Run:
    python train.py --save_dir 'the path leads to the saved result folder' 
### 3.3. Testing process
#### 3.3.1. Data
put .csv file in the test data folder
#### 3.3.2. Run data
    python predict.py
    
## 4. Result
![plot](./images/result.png)
## 5. Citation: TODO
