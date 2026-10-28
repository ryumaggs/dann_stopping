import numpy as np
from DANN_GPU import DANN
from mSDA import compute_msda_representation
from sklearn.datasets import load_svmlight_files
from sklearn import svm
import torch
import os
from amazon_util import load_amazon
from stopping import pairwise_weighted_minibatch_scores, load_all_network_dirs

def main():
    data_folder = './results_and_data/data/'     # where the datasets are

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

    for c in combinations:
        main_dir = './results_and_data/all_exp_histories:'+str(c)+'/'
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
            print(f"Directory '{main_dir}' created.")
        else:
            print(f"Directory '{main_dir}' already exists.")
        
        source_name = c[0]
        target_name = c[1]
        num_experiments = 15
        adversarial = True          # set to False to learn a standard NN
        msda = True
        save_every_epoch = True
        early_valid_stop = False
        COMPUTE_MATRICES = True

        hidden_layer_size = 50
        hidden_dims = []
        lambda_adapt = 0.1 if adversarial else 0.
        learning_rate = 0.001 if not msda else 0.0001
        maxiter = 50
        dropout=0.0
        device = 'cuda:1'

        print("Loading data...")
        xs_orig, ys_orig, xt, _, xtest, ytest = load_amazon(source_name, target_name, data_folder, verbose=True)
        
        if msda:
            xs_path, xt_path, xtest_path = ['%s/%s.%s_%s_msda.npy' % (data_folder, source_name, target_name, E)
                                            for E in ('source', 'target', 'test')]
            try:
                xs_msda = np.load(xs_path)
                xt_msda = np.load(xt_path)
                xtest_msda = np.load(xtest_path)
                print('mSDA representations loaded from disk')
            except Exception as e:
                print('Computing mSDA representations...')
                xs_msda, xt_msda, xtest_msda = compute_msda_representation(xs_orig, xt, xtest)
                np.save(xs_path, xs_msda)
                np.save(xt_path, xt_msda)
                np.save(xtest_path, xtest_msda)

            xs_orig, xt, xtest = xs_msda, xt_msda, xtest_msda
            
        for _ in range(num_experiments):
            run_id = np.random.randint(low=1,high=1e6)
            np.random.seed(run_id)
            torch.manual_seed(run_id)
            torch.cuda.manual_seed(run_id)

            idx = np.random.permutation(len(xs_orig))
            xs, ys = xs_orig[idx], ys_orig[idx]

            nb_valid = int(0.1 * len(ys))
            xv, yv = xs[-nb_valid:, :], ys[-nb_valid:]
            xs, ys = xs[0:-nb_valid, :], ys[0:-nb_valid]

            
            if True:
                print("Fit...")
                algo = DANN(lambda_adapt=lambda_adapt, 
                            hidden_layer_size=hidden_layer_size,
                            hidden_dims = hidden_dims,
                            learning_rate=learning_rate,
                            maxiter=maxiter, 
                            dropout=dropout,
                            device=device,
                            epsilon_init=None, 
                            seed=run_id, 
                            adversarial_representation=adversarial, 
                            verbose=True,)
                train_risk_history, val_risk_history = algo.fit(xs, ys, xt, xv, yv, 
                                                                save_every_epoch=save_every_epoch,
                                                                early_valid_stop=early_valid_stop,
                                                                reset_saves_dir=True,
                                                                run_id=run_id)

                print("Predict...")
                prediction_train = algo.predict(xs)
                prediction_valid = algo.predict(xv)
                prediction_test = algo.predict(xtest)

                print(algo.model)

                print('Training Risk   = %f' % np.mean(prediction_train != ys))
                print('Validation Risk = %f' % np.mean(prediction_valid != yv))
                print('Test Risk       = %f' % np.mean(prediction_test != ytest))

                #del algo #clear up some memory in case we need it

                val_early_stop_stats = np.array([algo.nb_iter, np.mean(prediction_train != ys),np.mean(prediction_valid != yv),np.mean(prediction_test != ytest)])
                np.savez(os.path.join(main_dir,'valEarlyStopStats_id:'+str(run_id)+'.npz'),s=val_early_stop_stats, t=train_risk_history, v=val_risk_history)
            
            if COMPUTE_MATRICES:
                #compute matrix for stopping then delete and reset saves folder (done automatically on next iteration)
                network_dict = load_all_network_dirs('./saves/',lambda_adapt=lambda_adapt,
                                                    data_shape=xs.shape[1],
                                                    num_classes=len(set(ys.tolist())),
                                                    device=device)
                xs_device = torch.tensor(xs, dtype=torch.float32, device=device)
                xt_device = torch.tensor(xt, dtype=torch.float32, device=device)
                xtest_device = torch.tensor(xtest, dtype=torch.float32, device=device)

                scores_fromhalf, testing_acc_dict, scores_rawAcc, scores_loss_one_minus = pairwise_weighted_minibatch_scores(
                    network_dict=network_dict,
                    X_src = xs_device,
                    X_tgt = xt_device,
                    X_tst = xtest_device,
                    y_src = ys,
                    y_tst = ytest,
                    device = device,
                    seed = None,
                    use_eval = True,
                ) 

                np.savez(os.path.join(main_dir,'fullConfMatrix_id:'+str(run_id)+'.npz'),m=scores_fromhalf, 
                                                                                        t=testing_acc_dict, 
                                                                                        n =scores_rawAcc,
                                                                                        o =  scores_loss_one_minus)

                if False:
                    exit(1)
                    print('==================================================================')

                    print('Computing PAD on DANN representation...')
                    pad_dann = compute_proxy_distance(algo.hidden_representation(xs), algo.hidden_representation(xt), verbose=True)
                    print('PAD on DANN representation = %f' % pad_dann)

                    print('==================================================================')

                    print('Computing PAD on original data...')
                    pad_original = compute_proxy_distance(xs, xt, verbose=True)
                    print('PAD on original data = %f' % pad_original)


def compute_proxy_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)


if __name__ == '__main__':
    main()

