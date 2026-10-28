import numpy as np
import os
import pickle
from stopping import exp3_ix_selfplay_2p0s, rescale_01
import matplotlib.pyplot as plt

def batch_stopping(histories_path, suffix):

    ids = [f.split(":")[1].split('.')[0] for f in os.listdir(histories_path) if 'Conf' in f]

    all_results = {}
    all_results['f,'] = []
    all_results['r,'] = []
    all_results['l,'] = []

    all_results['favg,'] = []
    all_results['ravg,'] = []
    all_results['lavg,'] = []

    all_results[',s'] = []
    all_results[',avg'] = []

    all_results['f,s'] = []
    all_results['f,avg'] = []
    all_results['favg,s'] = []
    all_results['favg,avg'] = []
    all_results['f,bir'] = []

    all_results['r,s'] = []
    all_results['r,avg'] = []
    all_results['ravg,s'] = []
    all_results['ravg,avg'] = []
    all_results['r,bir'] = []


    all_results['l,s'] = []
    all_results['l,avg'] = []
    all_results['lavg,s'] = []
    all_results['lavg,avg'] = []
    all_results['l,bir'] = []


    all_results['best_test_risk'] = []
    all_results['best_in_region'] = []

    min_val_from_hist = []
    for idd in ids:
        if idd != str(688127):
            continue
        print(idd)
        conf_matrix_path = histories_path + 'fullConfMatrix_id:'+idd+'.npz'
        tester = np.load(conf_matrix_path,allow_pickle=True)
        saved_val_stop_all = np.load(histories_path+'valEarlyStopStats_id:'+idd+'.npz')
        
        scores = tester['m']
        scores1 = tester['n']
        scores2 = tester['o']
        testing_acc_dict = tester['t'].item()
        
        saved_val_stop = saved_val_stop_all['s']
        val_history = saved_val_stop_all['v']
        min_val_from_hist.append(np.min(val_history).item())


        rep_start_idx = 110
        domain_labels_start_idx = 110
        

        rep_end_idx = scores.shape[1]
        domain_labels_end_idx = scores.shape[1]

        vhist_shortened = val_history[rep_start_idx:rep_end_idx]
        all_val_min_idx = np.where(vhist_shortened == vhist_shortened.min())[0]
        

        
        conf_matrix = scores[rep_start_idx:rep_end_idx,domain_labels_start_idx:domain_labels_end_idx]
        conf_matrix1 = scores1[rep_start_idx:rep_end_idx,domain_labels_start_idx:domain_labels_end_idx]
        conf_matrix2 = scores2[rep_start_idx:rep_end_idx,domain_labels_start_idx:domain_labels_end_idx]

        #conf_matrix1 = conf_matrix1 - np.min(conf_matrix1)
        #conf_matrix_rescaled = rescale_01(conf_matrix)
        #conf_matrix_rescaled1 = rescale_01(conf_matrix1)
        conf_matrix_rescaled2 = rescale_01(conf_matrix2)


        maxT = 300000

        K1 = rep_end_idx - rep_start_idx
        K2 = domain_labels_end_idx - domain_labels_start_idx
        #raw acc params
        c = 1
        eta1 = (1 / c) * np.sqrt(2 * np.log(K1 + 1) / (maxT * K1))
        gamma1 = eta1 / 2
        eta2 = (1 / c) * np.sqrt(2 * np.log(K2 + 1) / (maxT * K2))
        gamma2 = eta2 / 2


        out = exp3_ix_selfplay_2p0s(L=conf_matrix,T=maxT,
                                    eta1=eta1,gamma1=gamma1,
                                    eta2=eta2,gamma2=gamma2)

        K1 = rep_end_idx - rep_start_idx
        K2 = domain_labels_end_idx - domain_labels_start_idx
        #raw acc params
        c = 1
        eta1 = (1 / c) * np.sqrt(2 * np.log(K1 + 1) / (maxT * K1))
        gamma1 = eta1 / 2
        eta2 = (1 / c) * np.sqrt(2 * np.log(K2 + 1) / (maxT * K2))
        gamma2 = eta2 / 2

        out1 = exp3_ix_selfplay_2p0s(L=conf_matrix1,T=maxT, eta1 = eta1, gamma1=gamma1,
                                     eta2=eta2,gamma2=gamma2,
                                     return_history=True)
        
        history_arr = np.stack(out1["history"]['p1'])  # shape: (T, N) where T=timesteps, N=array length

        num_elements = history_arr.shape[1]

        fig, axes = plt.subplots(num_elements, 1, figsize=(10, 2 * num_elements), sharex=True, sharey=True)
        if num_elements == 1:
            axes = [axes]

        for idx, ax in enumerate(axes):
            ax.plot(history_arr[:, idx])
            ax.set_ylabel(f"Element {idx}")
            ax.set_xlabel("Time")


        out2 = exp3_ix_selfplay_2p0s(L=conf_matrix_rescaled2,T=maxT)
        # print(out['p1_avg'])
        if False:
            print(out['p1_avg'][:10])
            print(out['p2_avg'][:10])
            print("Single best action in hindsight: ")
            print("best rep stopping point: ", np.argmax(out['p1_avg']) + rep_start_idx)
            print("best domain_label stopping point: ", np.argmax(out['p2_avg']))
            print("best testing loss: ", min(testing_acc_dict.items(), key=lambda x: x[1]))
            print("selected stopping: ", testing_acc_dict[np.argmax(out['p1_avg']) + rep_start_idx])
            print("-------------------------------")
            print("")
            print("averaged testing loss")
            print(print("averaged: ", np.dot(out['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:]))))
            print("")
            print("-------------------------------------")
            print("")
            print("Early val stop")
            
            print('Training Risk   = %f' % saved_val_stop[1])
            print('Validation Risk = %f' % saved_val_stop[2])
            print('Test Risk       = %f' % saved_val_stop[3])
            print("early val stop point: ", saved_val_stop[0])
            print("retesting early val stop: ", testing_acc_dict[int(saved_val_stop[0])])
        
        print("---------------")
        print("top X: ")
        print(out['p1_avg'])
        print(out1['p1_avg'])

        print("selected stopping from Half: ", np.argmax(out['p1_avg']), testing_acc_dict[np.argmax(out['p1_avg']) + rep_start_idx])
        print("averaged from Half: ", np.dot(out['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])))

        print("selected stopping raw Acc: ", np.argmax(out1['p1_avg']), testing_acc_dict[np.argmax(out1['p1_avg']) + rep_start_idx])
        print("averaged raw Acc: ", np.dot(out1['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])))

        print("selected stopping raw Loss: ", np.argmax(out2['p1_avg']), testing_acc_dict[np.argmax(out2['p1_avg']) + rep_start_idx])
        print("averaged raw Loss: ", np.dot(out2['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])))

        print("best/early val: ", int(saved_val_stop[0]), testing_acc_dict[int(saved_val_stop[0])])
        print("all best val: ", np.mean([testing_acc_dict[i] for i in all_val_min_idx]))

        best_possible_idx, best_possible_text = min(testing_acc_dict.items(), key=lambda x: x[1])
        print("best possible: ", best_possible_idx, best_possible_text, best_possible_idx - rep_start_idx)
        print("diff from half: ", testing_acc_dict[np.argmax(out['p1_avg']) + rep_start_idx] - testing_acc_dict[int(saved_val_stop[0])])
        print("diff raw acc: ", testing_acc_dict[np.argmax(out1['p1_avg']) + rep_start_idx] - testing_acc_dict[int(saved_val_stop[0])])
        print("diff raw Loss: ", testing_acc_dict[np.argmax(out2['p1_avg']) + rep_start_idx] - testing_acc_dict[int(saved_val_stop[0])])

        print('best_val same region: ', all_val_min_idx[-1].item() +rep_start_idx, testing_acc_dict[all_val_min_idx[-1].item()+rep_start_idx])
        
        exit(1)
        all_results['best_in_region'].append(testing_acc_dict[all_val_min_idx[-1].item()+rep_start_idx])
        all_results['f,'].append((testing_acc_dict[np.argmax(out['p1_avg']) + rep_start_idx]).item())
        all_results['r,'].append((testing_acc_dict[np.argmax(out1['p1_avg']) + rep_start_idx]).item())
        all_results['l,'].append((testing_acc_dict[np.argmax(out2['p1_avg']) + rep_start_idx]).item())

        all_results['f,bir'].append((testing_acc_dict[np.argmax(out['p1_avg']) + rep_start_idx]).item() - testing_acc_dict[all_val_min_idx[-1].item()+rep_start_idx])
        all_results['r,bir'].append((testing_acc_dict[np.argmax(out1['p1_avg']) + rep_start_idx]).item() - testing_acc_dict[all_val_min_idx[-1].item()+rep_start_idx])
        all_results['l,bir'].append((testing_acc_dict[np.argmax(out2['p1_avg']) + rep_start_idx]).item() - testing_acc_dict[all_val_min_idx[-1].item()+rep_start_idx])

        all_results['favg,'].append(np.dot(out['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])).item())
        all_results['ravg,'].append(np.dot(out1['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])).item())
        all_results['lavg,'].append(np.dot(out2['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])).item())

        all_results[',s'].append(testing_acc_dict[int(saved_val_stop[0])].item())
        all_results[',avg'].append(np.mean([testing_acc_dict[i] for i in all_val_min_idx]).item())

        all_results['f,s'].append((testing_acc_dict[np.argmax(out['p1_avg']) + rep_start_idx] - testing_acc_dict[int(saved_val_stop[0])]).item())
        all_results['r,s'].append((testing_acc_dict[np.argmax(out1['p1_avg']) + rep_start_idx] - testing_acc_dict[int(saved_val_stop[0])]).item())
        all_results['l,s'].append((testing_acc_dict[np.argmax(out2['p1_avg']) + rep_start_idx] - testing_acc_dict[int(saved_val_stop[0])]).item())
        
        all_results['favg,s'].append((np.dot(out['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])) - testing_acc_dict[int(saved_val_stop[0])]).item())
        all_results['ravg,s'].append((np.dot(out1['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])) - testing_acc_dict[int(saved_val_stop[0])]).item())
        all_results['lavg,s'].append((np.dot(out2['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])) - testing_acc_dict[int(saved_val_stop[0])]).item())

        all_results['f,avg'].append((testing_acc_dict[np.argmax(out['p1_avg']) + rep_start_idx] - np.mean([testing_acc_dict[i] for i in all_val_min_idx])).item())
        all_results['r,avg'].append((testing_acc_dict[np.argmax(out1['p1_avg']) + rep_start_idx] - np.mean([testing_acc_dict[i] for i in all_val_min_idx])).item())
        all_results['l,avg'].append((testing_acc_dict[np.argmax(out2['p1_avg']) + rep_start_idx] - np.mean([testing_acc_dict[i] for i in all_val_min_idx])).item())
        
        all_results['favg,avg'].append((np.dot(out['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])) - testing_acc_dict[int(saved_val_stop[0])]).item())
        all_results['ravg,avg'].append((np.dot(out1['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])) - testing_acc_dict[int(saved_val_stop[0])]).item())
        all_results['lavg,avg'].append((np.dot(out2['p1_avg'],np.array(list(testing_acc_dict.values())[rep_start_idx:rep_end_idx])) - testing_acc_dict[int(saved_val_stop[0])]).item())
    
        all_results['best_test_risk'].append(min(testing_acc_dict.items(), key=lambda x: x[1]))

    #paired = sorted(zip(min_val_from_hist, diff_history), key=lambda x: x[0])
    #min_val_from_hist_sorted, diff_history_sorted = zip(*paired)
    if False:
        batch_results_folder = './results_and_data/batch_results:' + suffix + '/'
        if not os.path.exists(batch_results_folder):
            os.makedirs(batch_results_folder)
            print(f"Directory '{batch_results_folder}' created.")
        else:
            print(f"Directory '{batch_results_folder}' already exists.")

        with open(os.path.join(batch_results_folder,'all_results_'+str(rep_start_idx)+','+str(domain_labels_start_idx)+str(rep_end_idx)+','+str(domain_labels_end_idx)+'.pikl'), 'wb') as file:
            pickle.dump(all_results,file)


if __name__ == "__main__":
    
    if False:
        all_classes = ['books', 'dvd', 'kitchen', 'electronics']
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
    histories_base_path = './results_and_data/all_exp_histories_mnist/'
    batch_stopping(histories_base_path, '_mnist')

    

    