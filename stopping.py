import pickle
import numpy as np
import torch
import pandas as pd
from DANN_GPU import DANNModel, MNISTDANNModel
from typing import Dict, Hashable, Tuple, Optional, Any
from tqdm import tqdm
import torch.nn as nn
import os
from amazon_util import load_amazon

def pairwise_weighted_minibatch_scores(
    network_dict: dict,
    X_src:  torch.tensor,
    X_tgt: torch.tensor,
    X_tst: torch.tensor,
    y_src: torch.tensor,
    y_tst: torch.tensor,
    *,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    use_eval: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each pairing (a_key in A, b_key in B):
      - sample K minibatches (each of size minibatch_size) from X using weights A[a_key]
      - stack them into a single tensor of shape (K, minibatch_size, D)
      - run network B[b_key] on that tensor
      - reduce the output to a scalar (mean over all elements)
      - store the scalar in a numpy matrix: result[i, j] corresponds to (A_keys[i], B_keys[j])

    Parameters
    ----------
    A : dict[int, np.ndarray]
        Keys are ints; values are nonnegative weights over rows of X with shape (M,).
        They do not need to be normalized (we will normalize internally).
    B : dict[int, torch.nn.Module]
        Keys are ints; values are neural nets that accept input shape (batch=K, mini-batch, D).
    X : np.ndarray
        Dataset of shape (M, D).
    K : int
        Number of randomly sampled minibatches per A key.
    minibatch_size : int
        Size of each minibatch (default: 32).
    replace : bool
        Sample with replacement from X (default: True). If False, note that each minibatch
        is sampled without replacement independently (so duplicates across minibatches can still occur).
    device : torch.device | None
        Device for running the networks. If None, inferred from each network's parameters.
    seed : int | None
        Seed for reproducible sampling.
    use_eval : bool
        If True, temporarily puts each network in eval() during scoring.

    Returns
    -------
    scores : np.ndarray
        Matrix of shape (len(A), len(B)), dtype float64.
    a_keys : np.ndarray
        The A keys in the row order used for 'scores'.
    b_keys : np.ndarray
        The B keys in the column order used for 'scores'.
    """

    scores_dist_from_half = np.empty((len(network_dict.keys()), len(network_dict.keys())), dtype=np.float64)
    scores_raw_acc = np.empty((len(network_dict.keys()), len(network_dict.keys())), dtype=np.float64)
    scores_raw_src_loss = np.empty((len(network_dict.keys()), len(network_dict.keys())), dtype=np.float64)
    domain_criterion = nn.BCEWithLogitsLoss()
    test_set_results = {}
    #iterate over all weight strategies
    for i, ak in enumerate(network_dict.keys()):
        network_dict[ak].eval() #set eval mode for network

        #just pass the entire data set in as a single batch and see scores
        src_dlabels = torch.ones(X_src.shape[0],  1, device=device)
        tgt_dlabels = torch.zeros(X_tgt.shape[0], 1, device=device)

        #generate representation for every data point
        with torch.no_grad():
            h_s = network_dict[ak].feature_extractor(X_src)
            h_t = network_dict[ak].feature_extractor(X_tgt)
        #
        for j, bk in enumerate(network_dict.keys()):
            network_dict[bk].eval()
            with torch.no_grad():
                d_logits_src = network_dict[bk].domain_discriminator(h_s)
                d_logits_tgt = network_dict[bk].domain_discriminator(h_t)

                d_preds_src = (torch.sigmoid(d_logits_src) >= 0.5).float()
                d_preds_tgt = (torch.sigmoid(d_logits_tgt) >= 0.5).float()

                correct_src = (d_preds_src == src_dlabels).float()
                correct_tgt = (d_preds_tgt == tgt_dlabels).float()

                domain_acc = torch.cat([correct_src, correct_tgt]).mean().item()

                if False and domain_acc < 0.1:
                    print(i,j,domain_acc)
                    print(d_logits_src[:10].T)
                    print(torch.sigmoid(d_logits_src)[:10].T)
                    print(d_preds_src[:10].T)
                    print("")
                    print(d_logits_tgt[:10].T)
                    print(torch.sigmoid(d_logits_tgt)[:10].T)
                    print(d_preds_tgt[:10].T)
                    exit(1)

                #domain_loss_src = torch.mean((pred.flatten() == src_dlabels).float())#domain_criterion(d_logits_src, src_dlabels)
                domain_loss_src = domain_criterion(d_logits_src, src_dlabels).cpu().item()
                # Reduce to scalar: mean over all elements (covers (K, ...), any shape)
                scores_dist_from_half[i,j] = 2 * abs(domain_acc - 0.5)
                scores_raw_acc[i,j] = domain_acc
                scores_raw_src_loss[i,j] = 1 - domain_loss_src

            #get testing set results
            if j not in test_set_results:
                prediction_test = network_dict[bk].predict(X_tst).cpu().numpy()
                test_set_results[j] = np.mean(prediction_test != y_tst)
    
    return scores_dist_from_half, test_set_results, scores_raw_acc, scores_raw_src_loss

def rescale_01(M):
    M = np.asarray(M, dtype=float)
    m_min = M.min()
    m_max = M.max()

    if m_max == m_min:
        return np.zeros_like(M)

    return (M - m_min) / (m_max - m_min)

def exp3_ix_selfplay_2p0s(
    L: np.ndarray,
    T: int = 100000,
    *,
    eta1: Optional[float] = None,
    gamma1: Optional[float] = None,
    eta2: Optional[float] = None,
    gamma2: Optional[float] = None,
    seed: Optional[int] = None,
    return_history: bool = False,
) -> Dict[str, Any]:
    """
    Simultaneous no-regret learning for both players in a 2-player zero-sum matrix game
    using EXP3-IX (bandit feedback).

    Inputs
    ------
    L : (n1, n2) ndarray
        Loss matrix for Player 1. Must have entries in [0, 1].
        Player 2's loss is defined as: L2 = 1 - L (per your spec).
    T : int
        Number of rounds.

    EXP3-IX (losses)
    ---------------
    - Maintain weights w over actions.
    - Play action a ~ p where p = w / sum(w).
    - Observe loss l(a) (bandit feedback).
    - Form implicit-exploration estimator:
          \hat{l}_a = l(a) / (p_a + gamma)
          \hat{l}_i = 0 for i != a
    - Update:
          w_a <- w_a * exp(-eta * \hat{l}_a)

    Returns
    -------
    dict with:
      - "p1_avg": average action distribution over time for player 1 (n1,)
      - "p2_avg": average action distribution over time for player 2 (n2,)
      - "p1_last": final round's distribution for player 1
      - "p2_last": final round's distribution for player 2
      - (optional) "history": per-round sampled actions & losses
    """

    p1_history = []
    p2_history = []
    L = np.asarray(L, dtype=float)
    if L.ndim != 2:
        raise ValueError(f"L must be 2D (n1, n2). Got shape {L.shape}")
    if np.any(L < 0.0) or np.any(L > 1.0):
        raise ValueError("All entries of L must be in [0, 1].")

    n1, n2 = L.shape
    if T <= 0:
        raise ValueError("T must be > 0.")

    rng = np.random.default_rng(seed)

    # Reasonable defaults (tunable). Keep them small to avoid numerical issues.
    if eta1 is None:
        eta1 = np.sqrt(2.0 * np.log(max(n1, 2)) / (T * max(n1, 1)))
    if gamma1 is None:
        gamma1 = 0.5 * eta1

    if eta2 is None:
        eta2 = np.sqrt(2.0 * np.log(max(n2, 2)) / (T * max(n2, 1)))
    if gamma2 is None:
        gamma2 = 0.5 * eta2

    # Weights
    w1 = np.ones(n1, dtype=float)
    w2 = np.ones(n2, dtype=float)

    # Running average of distributions (time-average mixed strategy)
    p1_sum = np.zeros(n1, dtype=float)
    p2_sum = np.zeros(n2, dtype=float)

    # Optional history
    if return_history:
        a1_hist = np.empty(T, dtype=int)
        a2_hist = np.empty(T, dtype=int)
        l1_hist = np.empty(T, dtype=float)
        l2_hist = np.empty(T, dtype=float)

    def safe_probs(w: np.ndarray) -> np.ndarray:
        w = np.maximum(w, 1e-300)  # prevent zeros
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback: uniform
            return np.full_like(w, 1.0 / w.size, dtype=float)
        return w / s

    for t in tqdm(range(T)):
        # Current mixed strategies
        p1 = safe_probs(w1)
        p2 = safe_probs(w2)

        p1_history.append(p1)
        p2_history.append(p2)

        # Accumulate for average strategy
        p1_sum += p1
        p2_sum += p2

        # Sample actions
        a1 = rng.choice(n1, p=p1)
        a2 = rng.choice(n2, p=p2)

        # Realized losses (bandit feedback)
        l1 = L[a1, a2]
        l2 = 1.0 - l1  # per your spec for 2p0s

        # EXP3-IX implicit exploration estimators
        # (only the played action gets a nonzero estimate)
        est1 = l1 / (p1[a1] + gamma1)
        est2 = l2 / (p2[a2] + gamma2)

        # Weight updates (loss setting => negative exponent)
        w1[a1] *= np.exp(-eta1 * est1)
        w2[a2] *= np.exp(-eta2 * est2)

        if return_history:
            a1_hist[t] = a1
            a2_hist[t] = a2
            l1_hist[t] = l1
            l2_hist[t] = l2

    p1_last = safe_probs(w1)
    p2_last = safe_probs(w2)

    out: Dict[str, Any] = {
        "p1_avg": p1_sum / T,
        "p2_avg": p2_sum / T,
        "p1_last": p1_last,
        "p2_last": p2_last,
        "params": {
            "T": T,
            "eta1": float(eta1),
            "gamma1": float(gamma1),
            "eta2": float(eta2),
            "gamma2": float(gamma2),
        },
    }

    if return_history:
        out["history"] = {
            "a1": a1_hist,
            "a2": a2_hist,
            "l1": l1_hist,
            "l2": l2_hist,
            "p1": p1_history,
            "p2": p2_history,
        }

    return out

def load_all_network_dirs(main_folder,lambda_adapt,data_shape,num_classes,
                          device, image=False,
                          network_start=0,
                          network_end=None,):
    network_dict = {}
    if network_end is None:
        network_checkpoint_0 = torch.load(os.path.join(main_folder,"network_ckpt_"+str(network_start)+".pt"), map_location="cpu")
        network_end = network_checkpoint_0['total_epochs']
    for i in range(network_start,network_end,1):
        effective_lambda = lambda_adapt
        network_checkpoint = torch.load(os.path.join(main_folder,"network_ckpt_"+str(i)+".pt"), map_location="cpu")
        if image:
            model = MNISTDANNModel(lambda_adapt=effective_lambda)
        else:
            model = DANNModel(data_shape, 
                            network_checkpoint['hidden_layer_size'], 
                            hidden_dims=network_checkpoint['hidden_dims'],
                            dropout=0.25,
                            num_classes=num_classes, 
                            lambda_adapt=effective_lambda)
        model.load_state_dict(network_checkpoint['model_state_dict'])
        model = model.to(device)
        network_dict[i] = model
        network_dict[i].eval()
        #break
    return network_dict

if __name__ == "__main__":
    from amazon_util import load_mnist_to_mnistm

    device = 'cuda:0'
    main_folder = './saves/'
    adversarial = True
    lambda_adapt = 0.1 if adversarial else 0.

    print("Loading data...")
    MNIST_m_path = './results_and_data/data/mnist_m/'
    xs_orig, ys_orig, xt, _, xtest, ytest = load_mnist_to_mnistm(MNIST_m_path,
                                                                  K_source=None,
                                                                  K_target=None,
                                                                  verbose=False)
    print(xs_orig.shape, xt.shape, xtest.shape, ytest.shape)

    # shuffle and split off validation set
    idx = np.random.permutation(len(xs_orig))
    xs, ys = xs_orig[idx], ys_orig[idx]
    nb_valid = int(0.1 * len(ys))
    xs, ys = xs[0:-nb_valid], ys[0:-nb_valid]

    network_dict = load_all_network_dirs(main_folder,
                                         lambda_adapt=lambda_adapt,
                                         data_shape=xs.shape[1],
                                         num_classes=len(set(ys.tolist())),
                                         device=device,
                                         image=True,)

    xs_device    = torch.tensor(xs,    dtype=torch.float32, device=device)
    xt_device    = torch.tensor(xt,    dtype=torch.float32, device=device)
    xtest_device = torch.tensor(xtest, dtype=torch.float32, device=device)

    scores_fromhalf, testing_acc_dict, scores_rawAcc, scores_loss_one_minus = pairwise_weighted_minibatch_scores(
        network_dict=network_dict,
        X_src=xs_device,
        X_tgt=xt_device,
        X_tst=xtest_device,
        y_src=ys,
        y_tst=ytest,
        device=device,
        seed=None,
        use_eval=True,
    )
    main_dir = "./results_and_data/all_exp_histories_mnist"
    np.savez(os.path.join(main_dir,'fullConfMatrix_id:'+str(453377)+'fixed.npz'),m=scores_fromhalf, 
                                                                                    t=testing_acc_dict, 
                                                                                    n =scores_rawAcc,
                                                                                    o=scores_loss_one_minus)


    exit(1)

    scores = scores_fromhalf


    rep_start_idx = 10
    domain_labels_start_idx = 10
    conf_matrix = scores[rep_start_idx:,domain_labels_start_idx:]

    conf_matrix_rescaled = rescale_01(conf_matrix)
    conf_matrix_rescaled = conf_matrix_rescaled



    out = exp3_ix_selfplay_2p0s(L=conf_matrix_rescaled,T=315000)
    # print(out['p1_avg'])

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
    saved_val_stop_all = np.load('./saves/val_early_stop_stats.npz')
    saved_val_stop = saved_val_stop_all['s']
    print('Training Risk   = %f' % saved_val_stop[1])
    print('Validation Risk = %f' % saved_val_stop[2])
    print('Test Risk       = %f' % saved_val_stop[3])
    print("early val stop point: ", saved_val_stop[0])
    print("retesting early val stop: ", testing_acc_dict[int(saved_val_stop[0])])