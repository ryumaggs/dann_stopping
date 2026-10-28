import pickle
import numpy as np
if __name__ == "__main__":
    with open('./DEV_test.pikl', 'rb') as f:
        data = pickle.load(f)

    for key, value in data.items():
        print(f"{key}: {value}")
    idd = str(688127)
    histories_path = "./results_and_data/all_exp_histories_mnist/"
    conf_matrix_path = histories_path + 'fullConfMatrix_id:'+idd+'.npz'
    tester = np.load(conf_matrix_path,allow_pickle=True)
    saved_val_stop_all = np.load(histories_path+'valEarlyStopStats_id:'+idd+'.npz')
    
    scores = tester['m']
    scores1 = tester['n']
    scores2 = tester['o']
    testing_acc_dict = tester['t'].item()

    print(testing_acc_dict)

    # Smallest key-value pair in testing_acc_dict by value
    best_test_key = min(testing_acc_dict, key=testing_acc_dict.get)
    print(f"Best testing_acc_dict entry: key={best_test_key}, value={testing_acc_dict[best_test_key]}")

    # Key in data with lowest value, and its corresponding testing_acc_dict value
    best_dev_key = min(data, key=data.get)
    print(f"Best DEV key: {best_dev_key}, DEV value={data[best_dev_key]}, testing_acc={testing_acc_dict[best_dev_key]}")
