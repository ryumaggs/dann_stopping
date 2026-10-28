import pickle
import numpy as np
import os
from scipy import stats


if __name__ == "__main__":
    all_classes = ['books', 'electronics','dvd', 'kitchen']
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
    
    combinations = ['_mnist']
    for c in combinations:
        print("---------------------------------------------")
        print(c)
        print("")
        batch_results_folder = './results_and_data/batch_results:'+str(c)+'/'

        rep_start_idx = 100
        domain_labels_start_idx = 100

        rep_end_idx = 150
        domain_labels_end_idx = 150

        fname = 'all_results_'+str(rep_start_idx)+','+str(domain_labels_start_idx)+str(rep_end_idx)+','+str(domain_labels_end_idx)+'.pikl'
        with open(os.path.join(batch_results_folder,fname), 'rb') as file:
            all_results = pickle.load(file)


        keyss = ['f,s','r,s','l,s','f,avg','r,avg','l,avg', 'f,bir']

        for k in keyss:
            print(k, np.mean(all_results[k]), np.std(all_results[k]))
            t_stat, p_value = stats.ttest_1samp(all_results[k], popmean=0, alternative='less')
            print(t_stat, p_value)
            
            differences = all_results[k]
            mean_diff = np.mean(differences)
            cohen_d   = mean_diff / np.std(differences, ddof=1)
            ci        = stats.t.interval(0.95, df=len(differences)-1,
                                        loc=mean_diff,
                                        scale=stats.sem(differences))

            print(f"Mean difference: {mean_diff:.4f}")
            print(f"Cohen's d:       {cohen_d:.4f}")
            print(f"95% CI:          ({ci[0]:.4f}, {ci[1]:.4f})")
            print("")

        print("SOLO KEYS: ")
        solo_keys = ['f,','r,','l,',',s',',avg']
        for solo_k in solo_keys:
            print(solo_k, np.mean(all_results[k]))

        print("")


        print(all_results['best_test_risk'])


        t_stat, p_value = stats.ttest_rel(all_results['f,'], all_results[',s'], alternative='greater')
        print(t_stat, p_value)
        a = input("press enter to continue...")
        if a == 'q':
            exit(1)