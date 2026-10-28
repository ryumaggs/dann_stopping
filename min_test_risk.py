import numpy as np
import os
def batch_stopping(histories_path, suffix):
    mean_best = []
    min_val_from_hist = []
    ids = [f.split(":")[1].split('.')[0] for f in os.listdir(histories_path) if 'Conf' in f]
    for idd in ids:
        conf_matrix_path = histories_path + 'fullConfMatrix_id:'+idd+'.npz'
        tester = np.load(conf_matrix_path,allow_pickle=True)
        testing_acc_dict = tester['t'].item()
        mean_best.append(min(testing_acc_dict.items(), key=lambda x: x[1])[1])

        saved_val_stop_all = np.load(histories_path+'valEarlyStopStats_id:'+idd+'.npz')

        saved_val_stop = saved_val_stop_all['s']
        val_history = saved_val_stop_all['v']
        min_val_from_hist.append(testing_acc_dict[int(saved_val_stop[0])].item())
    

    print(suffix, np.mean(mean_best), np.mean(min_val_from_hist))

if __name__ == "__main__":
    all_classes = ['books', 'electronics', 'dvd', 'kitchen', ]
    combinations = []
    for source_name in all_classes:
        for target_name in all_classes:
            if source_name == target_name:
                continue
            if source_name == 'dvd' and target_name == 'electronics':
                continue
            #source_name = 'books'         # source domain: books, dvd, kitchen, or electronics
            #target_name = 'kitchen' # traget domain: books, dvd, kitchen, or electronics
            combinations.append((source_name, target_name))
    
    print(combinations)

    histories_base_path = './results_and_data/all_exp_histories:'

    for c in combinations:
        histories_path = histories_base_path+str(c)+'/'
        suffix = str(c)

        batch_stopping(histories_path, suffix)
