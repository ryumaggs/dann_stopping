import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from DANN_GPU import DANNModel
from amazon_util import load_amazon, load_svmlight_files, load_mnist_to_mnistm
import os
from stopping import load_all_network_dirs
from tqdm import tqdm
import pickle

def get_dev_risk(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1),rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta


def get_weight(source_feature, target_feature, validation_feature):
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    source_feature = source_feature.copy()
    target_feature = target_feature.copy()
    all_feature = np.concatenate((source_feature, target_feature))
    all_label = np.asarray([1] * N_s + [0] * N_t,dtype=np.int32)
    feature_for_train,feature_for_test, label_for_train,label_for_test = train_test_split(all_feature, all_label, train_size = 0.8)
    
    decays = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    val_acc = []
    domain_classifiers = []
    
    for decay in decays:
        domain_classifier = MLPClassifier(hidden_layer_sizes=(d, d, 2),activation='relu',alpha=decay)
        domain_classifier.fit(feature_for_train, label_for_train)
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test == output).astype(np.float32))
        val_acc.append(acc.item())
        domain_classifiers.append(domain_classifier)
        print('decay is %s, val acc is %s'%(decay, acc))
        
    index = val_acc.index(max(val_acc))
    
    print('val acc is')
    print(val_acc)

    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature)
    return domain_out[:,:1] / domain_out[:,1:] * N_s * 1.0 / N_t


def get_weight_image(source_features, target_features, validation_features, device,
                     decays=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
                     epochs=20, batch_size=64):
    """
    GPU version of get_weight operating on pre-extracted features.

    :param source_features:     [N_s, d] tensor on device (output of feature_extractor)
    :param target_features:     [N_t, d] tensor on device
    :param validation_features: [N_v, d] tensor on device
    :return: importance weights shape [N_v, 1]
    """
    N_s = source_features.shape[0]
    N_t = target_features.shape[0]
    d   = source_features.shape[1]

    all_features = torch.cat([source_features, target_features], dim=0)
    all_labels   = torch.cat([
        torch.ones(N_s, device=device),
        torch.zeros(N_t, device=device)
    ]).long()

    # 80/20 train/val split for selecting best decay
    n_total = all_features.shape[0]
    n_train = int(0.8 * n_total)
    perm    = torch.randperm(n_total, device=device)
    tr_idx, val_idx = perm[:n_train], perm[n_train:]
    X_tr, y_tr   = all_features[tr_idx], all_labels[tr_idx]
    X_val, y_val = all_features[val_idx], all_labels[val_idx]

    best_acc   = -1
    best_decay = decays[0]
    best_state = None

    for decay in decays:
        model     = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=decay)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(X_tr, y_tr)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val).argmax(dim=1)
            acc   = (preds == y_val).float().mean().item()

        print(f'decay={decay}, val acc={acc:.4f}')
        if acc > best_acc:
            best_acc   = acc
            best_decay = decay
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f'Best decay: {best_decay}, val acc: {best_acc:.4f}')

    best_model = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2)).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    with torch.no_grad():
        logits = best_model(validation_features)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()  # [N_v, 2]: col0=target, col1=source

    # importance weight = P(source) / P(target) * N_s / N_t
    return probs[:, 1:2] / probs[:, 0:1] * N_s / N_t


def load_mnist(K_source=None, K_target=None, verbose=False):
    MNIST_m_path = './results_and_data/data/mnist_m/'
    print("Loading data...")
    xs_orig, ys_orig, xt, _, xtest, ytest = load_mnist_to_mnistm(MNIST_m_path,
                                                                  K_source=K_source,
                                                                  K_target=K_target,
                                                                  verbose=verbose)
    
    idx = np.random.permutation(len(ys_orig))
    xs, ys = xs_orig[idx], ys_orig[idx]

    nb_valid = int(0.1 * len(ys))
    xv, yv = xs[-nb_valid:], ys[-nb_valid:]
    xs, ys = xs[:-nb_valid], ys[:-nb_valid]

    return xs, ys, xv, yv, xt, xtest, ytest



def load_amazon():
    data_folder = './results_and_data/data/'     # where the datasets are
    source_name = 'dvd'         # source domain: books, dvd, kitchen, or electronics
    target_name = 'electronics' # traget domain: books, dvd, kitchen, or electronics
    print("Loading data...")
    xs, ys, xt, _, xtest, ytest = load_amazon(source_name, target_name, data_folder, verbose=True)
    if msda:
        xs_path, xt_path, xtest_path = ['%s/%s.%s_%s_msda.npy' % (data_folder, source_name, target_name, E)
                                        for E in ('source', 'target', 'test')]
    try:
        xs_msda = np.load(xs_path)
        xt_msda = np.load(xt_path)
        xtest_msda = np.load(xtest_path)
        print('mSDA representations loaded from disk')
        xs, xt, xtest = xs_msda, xt_msda, xtest_msda
    except:
        assert NotImplementedError, "MSDA reps should have already been computed during training"
    
    idx = np.random.permutation(len(ys))
    xs, ys = xs[idx], ys[idx]

    nb_valid = int(0.1 * len(ys))
    xv, yv = xs[-nb_valid:], ys[-nb_valid:]
    xs, ys = xs[:-nb_valid], ys[:-nb_valid]

    return xs, ys, xv, yv, xt, xtest, ytest

if __name__ == "__main__":
    device = torch.device('cuda:0')
    main_folder = './saves_seed:688127_MNIST/'
    
    adversarial = True          # set to False to learn a standard NN
    msda = True
    save_every_epoch = True
    early_valid_stop = False
    run_id = np.random.randint(low=1, high=int(1e6))

    hidden_layer_size = 50
    lambda_adapt = 0.1 if adversarial else 0.
    learning_rate = 0.001 if not msda else 0.0001
    maxiter = 200
    saved_every=1

    seed = 688127
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    xs, ys, xv, yv, xt, xtest, ytest = load_mnist()
    network_start = 110
    network_end = 150
    all_results = {}
    print("Loading networks...")
    network_dict = load_all_network_dirs(main_folder,lambda_adapt,
                          xs.shape[1],len(set(ys.tolist())),
                          device, image=True,
                          network_start=network_start,
                          network_end=network_end,)

    
    risk_arr = {}
    counter = network_start
    print("Beginning DEV score computation")
    for ak in tqdm(list(network_dict.keys())):
        #sample the 3 features
        X_src = torch.tensor(xs,device=device,dtype=torch.float)
        X_tgt = torch.tensor(xtest,device=device,dtype=torch.float)
        X_val = torch.tensor(xv,device=device,dtype=torch.float)
        with torch.no_grad():
            h_s = network_dict[ak].feature_extractor(X_src)
            h_t = network_dict[ak].feature_extractor(X_tgt)
            h_v = network_dict[ak].feature_extractor(X_val)
            val_preds = network_dict[ak].predict(X_val).cpu().numpy()

        error = np.expand_dims((val_preds != yv).astype(float),1)

        src_features = h_s
        tgt_features = h_t
        val_features = h_v


        #pass to get weight
        #dev_weights = get_weight(src_features, tgt_features, val_features)
        dev_weights = get_weight_image(source_features=src_features, 
                                       target_features=tgt_features, 
                                       validation_features=val_features, 
                                       device=device,
                                       decays=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])


        #get devweight calculation
        risk = get_dev_risk(dev_weights, error)

        risk_arr[counter] = risk
        counter += 1

        with open("./DEV_test.pikl", 'wb') as file:
            pickle.dump(risk_arr, file)

        
