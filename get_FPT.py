from utils.tools import predict_time, convert_to_image
from train import parse_opt
from os.path import join 

opt = parse_opt()
if opt.type == 'PHM':
    # Link of original data ==================================================================================
    train_dir = join(opt.main_dir_colab, 'PHM_data/Learning_set')
    test_dir = join(opt.main_dir_colab, 'PHM_data/Test_set')

    # Load 1d data ==================================================================================
    print('Load PHM data')
    # Train data
    Bearing1_1 = convert_to_image(join(train_dir, 'Bearing1_1'), opt, '1d', None, 'PHM')
    Bearing1_2 = convert_to_image(join(train_dir, 'Bearing1_2'), opt, '1d', None, 'PHM')
    Bearing2_1 = convert_to_image(join(train_dir, 'Bearing2_1'), opt, '1d', None, 'PHM')
    Bearing2_2 = convert_to_image(join(train_dir, 'Bearing2_2'), opt, '1d', None, 'PHM')
    Bearing3_1 = convert_to_image(join(train_dir, 'Bearing3_1'), opt, '1d', None, 'PHM')
    Bearing3_2 = convert_to_image(join(train_dir, 'Bearing3_2'), opt, '1d', None, 'PHM')

    # Test data
    Bearing1_3 = convert_to_image(join(test_dir, 'Bearing1_1'), opt, '1d', None, 'PHM')
    Bearing1_4 = convert_to_image(join(test_dir, 'Bearing1_2'), opt, '1d', None, 'PHM')
    Bearing1_5 = convert_to_image(join(test_dir, 'Bearing2_1'), opt, '1d', None, 'PHM')
    Bearing1_6 = convert_to_image(join(test_dir, 'Bearing2_2'), opt, '1d', None, 'PHM')
    Bearing1_7 = convert_to_image(join(test_dir, 'Bearing3_1'), opt, '1d', None, 'PHM')

    Bearing2_3 = convert_to_image(join(test_dir, 'Bearing1_1'), opt, '1d', None, 'PHM')
    Bearing2_4 = convert_to_image(join(test_dir, 'Bearing1_2'), opt, '1d', None, 'PHM')
    Bearing2_5 = convert_to_image(join(test_dir, 'Bearing2_1'), opt, '1d', None, 'PHM')
    Bearing2_6 = convert_to_image(join(test_dir, 'Bearing2_2'), opt, '1d', None, 'PHM')
    Bearing2_7 = convert_to_image(join(test_dir, 'Bearing3_1'), opt, '1d', None, 'PHM')

    Bearing3_3 = convert_to_image(join(test_dir, 'Bearing3_3'), opt, '1d', None, 'PHM')

    data = {'Bearing1_1': Bearing1_1, 'Bearing1_2': Bearing1_2, 'Bearing1_3': Bearing1_3, 'Bearing1_4': Bearing1_4,
            'Bearing1_5': Bearing1_5, 'Bearing1_6': Bearing1_6, 'Bearing1_7': Bearing1_7, 'Bearing2_1': Bearing2_1, 
            'Bearing2_2': Bearing2_2, 'Bearing2_3': Bearing2_3, 'Bearing2_4': Bearing2_4, 'Bearing2_5': Bearing2_5,
            'Bearing2_6': Bearing2_6, 'Bearing2_7': Bearing2_7, 'Bearing3_1': Bearing3_1, 'Bearing3_2': Bearing3_2, 'Bearing3_3': Bearing3_3}
    for bearing in data:
        fpt = predict_time(data[bearing]['x'])
        print(f'---{bearing}---')
        d_shape = data[bearing]['x'].shape
        print(f'Shape: {d_shape} \t FPT: {fpt}\n')
else:
    print('Load XJTU data')
    main_dir_colab = join(opt.main_dir_colab, 'XJTU_data/XJTU-SY_Bearing_Datasets')
    #---------------------------------------------------------------------------------
    Bearing1_1_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_1')
    Bearing1_1 = convert_to_image(Bearing1_1_path, opt, '1d', None, 'XJTU')

    Bearing1_2_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_2')
    Bearing1_2 = convert_to_image(Bearing1_2_path, opt, '1d', None, 'XJTU')
    
    Bearing1_3_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_3')
    Bearing1_3 = convert_to_image(Bearing1_3_path, opt, '1d', None, 'XJTU')

    Bearing1_4_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_4')
    Bearing1_4 = convert_to_image(Bearing1_4_path, opt, '1d', None, 'XJTU')

    Bearing1_5_path = join(main_dir_colab, '35Hz12kN', 'Bearing1_5')
    Bearing1_5 = convert_to_image(Bearing1_5_path, opt, '1d', None, 'XJTU')

    #---------------------------------------------------------------------------------
    Bearing2_1_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_1')
    Bearing2_1 = convert_to_image(Bearing2_1_path, opt, '1d', None, 'XJTU')

    Bearing2_2_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_2')
    Bearing2_2 = convert_to_image(Bearing2_2_path, opt, '1d', None, 'XJTU')
    
    Bearing2_3_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_3')
    Bearing2_3 = convert_to_image(Bearing2_3_path, opt, '1d', None, 'XJTU')

    Bearing2_4_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_4')
    Bearing2_4 = convert_to_image(Bearing2_4_path, opt, '1d', None, 'XJTU')

    Bearing2_5_path = join(main_dir_colab, '37.5Hz11kN', 'Bearing2_5')
    Bearing2_5 = convert_to_image(Bearing2_5_path, opt, '1d', None, 'XJTU')

    #---------------------------------------------------------------------------------
    Bearing3_1_path = join(main_dir_colab, '40Hz10kN', 'Bearing3_1')
    Bearing3_1 = convert_to_image(Bearing3_1_path, opt, '1d', None, 'XJTU')

    Bearing3_2_path = join(main_dir_colab, '40Hz10kN', 'Bearing3_2')
    Bearing3_2 = convert_to_image(Bearing3_2_path, opt, '1d', None, 'XJTU')
    
    Bearing3_3_path = join(main_dir_colab, '40Hz10kN', 'Bearing3_3')
    Bearing3_3 = convert_to_image(Bearing3_3_path, opt, '1d', None, 'XJTU')

    Bearing3_4_path = join(main_dir_colab, '40Hz10kN', 'Bearing3_4')
    Bearing3_4 = convert_to_image(Bearing3_4_path, opt, '1d', None, 'XJTU')

    Bearing3_5_path = join(main_dir_colab, '40Hz10kN', 'Bearing3_5')
    Bearing3_5 = convert_to_image(Bearing3_5_path, opt, '1d', None, 'XJTU')

    data = {'Bearing1_1': Bearing1_1, 'Bearing1_2': Bearing1_2, 'Bearing1_3': Bearing1_3, 'Bearing1_4': Bearing1_4,
            'Bearing1_5': Bearing1_5, 'Bearing2_1': Bearing2_1, 'Bearing2_2': Bearing2_2, 'Bearing2_3': Bearing2_3, 
            'Bearing2_4': Bearing2_4, 'Bearing2_5': Bearing2_5, 'Bearing3_1': Bearing3_1, 'Bearing3_2': Bearing3_2, 
            'Bearing3_3': Bearing3_3, 'Bearing3_4': Bearing3_4, 'Bearing3_5': Bearing3_5}

    for bearing in data:
        fpt = predict_time(data[bearing]['x'])
        print(f'---{bearing}---')
        d_shape = data[bearing]['x'].shape
        print(f'Shape: {d_shape} \t FPT: {fpt}\n')