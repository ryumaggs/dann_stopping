import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import torch.nn.functional as F
import shutil
from ComplexNet import ComplexNeuralNetwork
# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """
    Forward pass is identity.
    Backward pass negates the gradient and scales by lambda_adapt.
    This replaces the manual sign-flip in the original numpy implementation.
    """
    @staticmethod
    def forward(ctx, x, lambda_adapt):
        ctx.save_for_backward(torch.tensor(lambda_adapt))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_adapt,) = ctx.saved_tensors
        return -lambda_adapt * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_adapt=1.0):
        super().__init__()
        self.lambda_adapt = lambda_adapt

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_adapt)


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------

def _build_layers(dims: list, dropout=0.0) -> nn.Sequential:
    """
    Given a list of layer widths [d0, d1, d2, ..., dN], build a Sequential
    of (Linear -> ReLU) blocks for every consecutive pair.

    Example: dims=[512, 256, 128] produces:
        Linear(512->256) -> ReLU -> Linear(256->128) -> ReLU

    He (Kaiming) initialisation is used for each Linear weight, which is the
    correct pairing for ReLU activations. Xavier/Glorot (used in the original
    numpy code) is designed for sigmoid/tanh and underestimates the required
    variance for ReLU networks.
    """
    layers = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        linear = nn.Linear(in_dim, out_dim)
        nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))

    # final layer: no activation
    linear = nn.Linear(dims[-2], dims[-1])
    nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
    nn.init.zeros_(linear.bias)
    layers.append(linear)

    return nn.Sequential(*layers)

class FeatureExtractor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 hidden_dims: list = None, dropout: float = 0.0):
        super().__init__()
        hidden_dims = hidden_dims or []

        # hidden_dims are intermediate widths between input and the
        # final hidden_size output — hidden_size is always the last dim
        all_dims = [input_size] + hidden_dims + [hidden_size]
        self.layers = _build_layers(all_dims, dropout)

        # output_size is always hidden_size regardless of intermediate dims
        self.output_size = hidden_size

    def forward(self, x):
        h = self.layers(x)
        #return F.normalize(h, p=2, dim=1)
        return h

class LabelPredictor(nn.Module):
    """
    Network 2: maps hidden representation -> class logits.
    Corresponds to (V, c) in the original numpy code.
    Receives gradients normally from cross-entropy loss.

    Parameters
    ----------
    hidden_size : int
        Input dimensionality — must equal FeatureExtractor.output_size.
    num_classes : int
        Number of output classes.
    hidden_dims : list[int], optional
        Extra hidden layers inserted between the input and the final
        classification layer.
        e.g. hidden_size=64, hidden_dims=[32], num_classes=2:
            h -> Linear(64->32) -> ReLU -> Linear(32->2)  <- raw logits
        Default [] is a single linear projection (original behaviour).
    """
    def __init__(self, hidden_size: int, num_classes: int,
                 hidden_dims: list = None, dropout: float = 0):
        super().__init__()
        hidden_dims = hidden_dims or []

        # Optional hidden stack — Identity if no extra dims requested
        all_hidden = [hidden_size] + hidden_dims
        self.hidden_layers = _build_layers(all_hidden, dropout) if hidden_dims else nn.Identity()

        # Final projection to class logits — no activation because
        # CrossEntropyLoss applies log-softmax internally
        final_in = hidden_dims[-1] if hidden_dims else hidden_size
        self.output_layer = nn.Linear(final_in, num_classes)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, h):
        return self.output_layer(self.hidden_layers(h))

class DomainDiscriminator(nn.Module):
    """
    Network 3: maps hidden representation -> domain logit (source=1, target=0).
    Corresponds to (U, d) in the original numpy code.

    The GradientReversalLayer sits at the front so that all gradients —
    including those passing through any extra hidden layers — are negated
    before flowing back into the feature extractor.

    Parameters
    ----------
    hidden_size : int
        Input dimensionality — must equal FeatureExtractor.output_size.
    lambda_adapt : float
        Gradient reversal scale factor.
    hidden_dims : list[int], optional
        Extra hidden layers inserted between the GRL output and the final
        binary logit.
        e.g. hidden_size=64, hidden_dims=[32]:
            h -> GRL -> Linear(64->32) -> ReLU -> Linear(32->1)
        Default [] is a single linear projection (original behaviour).
    """
    def __init__(self, hidden_size: int, lambda_adapt: float = 1.0,
                 hidden_dims: list = None,dropout: float = 0):
        super().__init__()
        hidden_dims = hidden_dims or []

        self.grl = GradientReversalLayer(lambda_adapt)

        # Hidden stack lives after the GRL so reversed gradients propagate
        # through the full depth back into the feature extractor
        all_hidden = [hidden_size] + hidden_dims
        self.hidden_layers = _build_layers(all_hidden,dropout) if hidden_dims else nn.Identity()

        # Final binary output — raw logit for BCEWithLogitsLoss
        final_in = hidden_dims[-1] if hidden_dims else hidden_size
        self.output_layer = nn.Linear(final_in, 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def set_lambda(self, lambda_adapt: float):
        self.grl.lambda_adapt = lambda_adapt

    def forward(self, h):
        h_reversed = self.grl(h)
        return self.output_layer(self.hidden_layers(h_reversed))

class FeatureExtractorComplex(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 hidden_dims: list = None, dropout: float = 0.0):
        super().__init__()
        hidden_dims = hidden_dims or []

        self.network = ComplexNeuralNetwork(
        layer_sizes=hidden_dims,
        activation=nn.ReLU(),
        use_bias=True
        )

    def forward(self, x):
        h = self.network(x)
        #return F.normalize(h, p=2, dim=1)
        return h

# ---------------------------------------------------------------------------
# Image Sub-networks
# ---------------------------------------------------------------------------

def build_mnist_feature_extractor() -> nn.Sequential:
    """
    MNIST Feature Extractor (Ganin et al. 2016, Figure 4a)

    conv 5x5, 32 maps -> ReLU -> max-pool 2x2 stride 2
    conv 5x5, 48 maps -> ReLU -> max-pool 2x2 stride 2

    Input:  [B, 3, 32, 32]
    Output: [B, 48*5*5] = [B, 1200]  (flattened)
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(32, 48, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Flatten()
    )


def build_mnist_label_predictor() -> nn.Sequential:
    """
    MNIST Label Predictor (Ganin et al. 2016, Figure 4a)

    FC 100 -> ReLU -> FC 100 -> ReLU -> FC 10 -> (Softmax)
    
    Softmax is omitted here; use CrossEntropyLoss which applies
    log-softmax internally.

    Input:  [B, 1200]
    Output: [B, 10]
    """
    return nn.Sequential(
        nn.Linear(1200, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )


def build_mnist_domain_classifier(lambda_adapt: float = 1.0) -> nn.Sequential:
    """
    MNIST Domain Classifier (Ganin et al. 2016, Figure 4a)

    GRL -> FC 100 -> ReLU -> FC 1 -> (Logistic)

    Logistic/sigmoid is omitted here; use BCEWithLogitsLoss which applies
    sigmoid internally.

    Input:  [B, 1200]
    Output: [B, 1]
    """
    return nn.Sequential(
        GradientReversalLayer(lambda_adapt),
        nn.Linear(1200, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )


# ---------------------------------------------------------------------------
# Full DANN model
# ---------------------------------------------------------------------------

class DANNModel(nn.Module):
    """
    Combines all three sub-networks into one nn.Module so that a single
    optimizer can manage all parameters.
    """
    def __init__(self, input_size: int, 
                 hidden_size: int,
                 hidden_dims,
                 dropout: float,
                 num_classes: int, 
                 lambda_adapt: float,
                 complex_feature: bool = False,
                 ):
        super().__init__()
        if not complex_feature:
            self.feature_extractor  = FeatureExtractor(input_size, hidden_size, hidden_dims=hidden_dims, dropout=dropout)
        else:
            self.feature_extractor  = FeatureExtractorComplex(input_size, hidden_size, hidden_dims=hidden_dims, dropout=dropout)
        self.label_predictor    = LabelPredictor(hidden_size, num_classes, hidden_dims=hidden_dims, dropout=dropout)
        self.domain_discriminator = DomainDiscriminator(hidden_size, lambda_adapt,  hidden_dims=hidden_dims, dropout=dropout)

    def forward(self, x):
        h        = self.feature_extractor(x)
        y_logits = self.label_predictor(h)
        d_logits = self.domain_discriminator(h)
        return y_logits, d_logits

    def predict(self, x):
        h        = self.feature_extractor(x)
        y_logits = self.label_predictor(h)
        return torch.argmax(y_logits, dim=1)

    def hidden_representation(self, x):
        return self.feature_extractor(x)

# ---------------------------------------------------------------------------
# Full IMAGE DANN model
# ---------------------------------------------------------------------------

class MNISTDANNModel(nn.Module):
    """
    MNIST-specific DANN combining the CNN feature extractor with the
    label predictor and domain classifier described in Ganin et al. 2016
    Figure 4a.

    Unlike the shallow DANNModel, architecture is fixed — no hidden_dims,
    hidden_size, input_size, or complex_feature arguments, as these are
    all determined by the MNIST CNN architecture.

    Parameters
    ----------
    lambda_adapt : float
        Gradient reversal scale factor passed to the domain classifier.
    """
    def __init__(self, lambda_adapt: float = 1.0):
        super().__init__()
        self.feature_extractor    = build_mnist_feature_extractor()
        self.label_predictor      = build_mnist_label_predictor()
        self.domain_discriminator = build_mnist_domain_classifier(lambda_adapt)

    def forward(self, x):
        h        = self.feature_extractor(x)
        y_logits = self.label_predictor(h)
        d_logits = self.domain_discriminator(h)
        return y_logits, d_logits

    def predict(self, x):
        h        = self.feature_extractor(x)
        y_logits = self.label_predictor(h)
        return torch.argmax(y_logits, dim=1)

    def hidden_representation(self, x):
        return self.feature_extractor(x)

    def set_lambda(self, lambda_adapt: float):
        """
        Update the gradient reversal scale factor during training.
        Allows the lambda schedule from the paper to be applied externally
        without rebuilding the model.
        """
        self.domain_discriminator[0].lambda_adapt = lambda_adapt


# ---------------------------------------------------------------------------
# DANN trainer  —  API matches the original numpy DANN class
# ---------------------------------------------------------------------------

class DANN:
    """
    Domain Adversarial Neural Network for classification.

    Drop-in replacement for the original numpy implementation, now using
    PyTorch with GPU acceleration where available.

    Parameters
    ----------
    learning_rate : float
        SGD learning rate.
    hidden_layer_size : int
        Number of units in the shared hidden layer.
    lambda_adapt : float
        Weight on the domain adaptation loss. 0 disables adaptation.
    maxiter : int
        Maximum number of training epochs.
    adversarial_representation : bool
        If False, the GRL lambda is forced to 0 so domain gradients do not
        affect the feature extractor. Matches the ablation flag in the original.
    batch_size : int
        Mini-batch size. The original code used online SGD (batch_size=1).
        Larger values exploit GPU parallelism.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress during training.
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 hidden_layer_size: int = 25,
                 hidden_dims = None,
                 lambda_adapt: float = 1.0,
                 maxiter: int = 200,
                 epsilon_init=None,           # kept for API compatibility, unused
                 adversarial_representation: bool = True,
                 batch_size: int = 1,
                 dropout: int = 0,
                 device: str = None,
                 seed: int = 12342,
                 verbose: bool = False,
                 complex_feature = False,
                 image_dataset = False):

        self.learning_rate              = learning_rate
        self.hidden_layer_size          = hidden_layer_size
        self.lambda_adapt               = lambda_adapt if lambda_adapt not in (None, False) else 0.0
        self.maxiter                    = maxiter
        self.adversarial_representation = adversarial_representation
        self.batch_size                 = batch_size
        self.seed                       = seed
        self.verbose                    = verbose
        self.hidden_dims                = hidden_dims
        self.dropout                    = dropout
        self.complex_feature            = complex_feature
        self.image_dataset              = image_dataset

        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if self.verbose:
            print(f'[DANN] using device: {self.device}')

        self.model      = None
        self.nb_iter    = 0
        self.valid_risk = 2.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, X, dtype=torch.float32):
        return torch.tensor(X, dtype=dtype, device=self.device)

    def _numpy_labels(self, Y):
        """Accept either numpy array or list; return numpy int array."""
        return np.asarray(Y, dtype=np.int64)

    def _build_model(self, input_size: int, num_classes: int):
        torch.manual_seed(self.seed)
        effective_lambda = self.lambda_adapt if self.adversarial_representation else 0.0
        model = DANNModel(input_size, 
                          self.hidden_layer_size, 
                          self.hidden_dims, 
                          self.dropout,
                          num_classes, 
                          effective_lambda)
        return model.to(self.device)
    
    def _build_model_MNIST(self):
        print("hit 2")
        torch.manual_seed(self.seed)
        effective_lambda = self.lambda_adapt if self.adversarial_representation else 0.0
        model = MNISTDANNModel(effective_lambda)
        return model.to(self.device)


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, Y, X_adapt,
            X_valid=None, Y_valid=None,
            X_test = None, Y_test = None,
            do_random_init: bool = True,
            save_every_epoch=False,
            early_valid_stop=True,
            reset_saves_dir=False,
            run_id=0):
        """
        Train the DANN.

        Parameters
        ----------
        X       : array (n_source, n_features)  — labeled source data
        Y       : array (n_source,)             — source labels (integer class indices)
        X_adapt : array (n_target, n_features)  — unlabeled target data
        X_valid : array (n_valid, n_features)   — optional validation features
        Y_valid : array (n_valid,)              — optional validation labels
        do_random_init : bool                   — reinitialise weights if True

        returns a series of relevant histories
        training_risk history
        val_risk history

        """
        writer = SummaryWriter(comment='run_id:'+str(run_id))
        
        main_dir = './saves/'
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
            print(f"Directory '{main_dir}' created.")
        else:
            print(f"Directory '{main_dir}' already exists.")
            if reset_saves_dir:
                print("Resetting Saves dir")
                shutil.rmtree(main_dir)
                os.makedirs(main_dir)
            

        Y = self._numpy_labels(Y)
        nb_features  = X.shape[1]
        num_classes  = len(set(Y.tolist()))
        print(nb_features, num_classes)

        if self.verbose:
            print(f'[DANN] source={X.shape}, target={X_adapt.shape}, '
                  f'classes={num_classes}, lambda={self.lambda_adapt}')

        # Build or reuse model
        if not self.image_dataset:
            if do_random_init or self.model is None:
                self.model = self._build_model(nb_features, num_classes)
        elif self.image_dataset:
            if do_random_init or self.model is None:
                self.model = self._build_model_MNIST()

        # Loss functions
        label_criterion  = nn.CrossEntropyLoss()
        domain_criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate,
                              momentum=0.9, weight_decay=1e-4)
        # Convert all data to tensors once (stays on GPU for the whole run)
        Xs_t  = self._to_tensor(X)
        Ys_t  = self._to_tensor(Y, dtype=torch.long)
        Xt_t  = self._to_tensor(X_adapt)

        # Domain labels: source=1, target=0
        source_domain_labels = torch.ones(self.batch_size, 1, device=self.device)
        target_domain_labels = torch.zeros(self.batch_size, 1, device=self.device)

        # DataLoaders for source and target — target has no labels
        source_dataset = TensorDataset(Xs_t, Ys_t)
        target_dataset = TensorDataset(Xt_t)

        source_loader = DataLoader(source_dataset, batch_size=self.batch_size,
                                   shuffle=True,  drop_last=True)
        target_loader = DataLoader(target_dataset, batch_size=self.batch_size,
                                   shuffle=True,  drop_last=True)

        # Validation tensors (optional)
        if X_valid is not None:
            Xv_t = self._to_tensor(X_valid)
            Yv_t = self._to_tensor(self._numpy_labels(Y_valid), dtype=torch.long)

        if X_test is not None:
            Xt_t = self._to_tensor(X_test)
            Yt_t = self._to_tensor(self._numpy_labels(Y_test), dtype=torch.long)

        best_valid_risk = 2.0
        best_weights    = None
        continue_until  = 30

        training_risk_history = []
        val_risk_history = []
        test_risk_history = []
        for epoch in tqdm(range(self.maxiter)):
            self.model.train()

            # Training progress p in [0, 1]
            p = epoch / self.maxiter

            # Learning rate annealing: µp = µ0 / (1 + α·p)^β  (Ganin et al.)
            lr_p = self.learning_rate / ((1 + 10 * p) ** 0.75)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_p

            # Lambda ramp: λp = 2/(1 + exp(-γ·p)) - 1, γ=10  (starts at 0, ends at 1)
            lambda_p = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
            self.model.set_lambda(lambda_p)

            # Zip source and target loaders; stop at the shorter one
            epoch_label_loss = 0
            epoch_domain_loss_src = 0
            epoch_domain_loss_tgt = 0
            counter = 0
            domain_perf = {}
            domain_perf['src'] = [0,0]
            domain_perf['tgt'] = [0,0]
            domain_correct = 0
            domain_total = 0

            for (x_s, y_s), (x_t,) in zip(source_loader, target_loader):
                bs = min(x_s.size(0), x_t.size(0))
                x_s, y_s, x_t = x_s[:bs], y_s[:bs], x_t[:bs]
                optimizer.zero_grad()

                # ---- Source forward pass ----
                h_s          = self.model.feature_extractor(x_s)
                y_logits     = self.model.label_predictor(h_s)
                d_logits_src = self.model.domain_discriminator(h_s)

                label_loss      = label_criterion(y_logits, y_s)
                domain_loss_src = domain_criterion(d_logits_src, source_domain_labels)

                # ---- Target forward pass (no label loss) ----
                h_t          = self.model.feature_extractor(x_t)
                d_logits_tgt = self.model.domain_discriminator(h_t)
                domain_loss_tgt = domain_criterion(d_logits_tgt, target_domain_labels)

                #if torch.nn.functional.sigmoid(d_logits_tgt.detach()).item() < 0.5:
                #    domain_perf['tgt'][0] += 1
                #else:
                #    domain_perf['tgt'][1] += 1
               # 
                #if torch.nn.functional.sigmoid(d_logits_src.detach()).item() >= 0.5:
                #    domain_perf['src'][0] += 1
                #else:
                #    domain_perf['src'][1] += 1

                # ---- Domain accuracy ----
                with torch.no_grad():
                    src_preds = (torch.sigmoid(d_logits_src) >= 0.5).float()
                    tgt_preds = (torch.sigmoid(d_logits_tgt) < 0.5).float()
                    domain_correct += src_preds.sum().item() + tgt_preds.sum().item()
                    domain_total += bs * 2

                # ---- Combined loss ----
                # The GRL inside DomainDiscriminator already negates the domain
                # gradient when it flows back into the feature extractor, so we
                # simply add all losses here — no manual sign manipulation needed.
                total_loss = label_loss + domain_loss_src + domain_loss_tgt

                epoch_label_loss += label_loss.detach().item()
                epoch_domain_loss_src += domain_loss_src.detach().item()
                epoch_domain_loss_tgt += domain_loss_tgt.detach().item()
                counter += x_s.shape[0]

                total_loss.backward()
                optimizer.step()

                #print(domain_loss_src.detach().item(), d_logits_tgt.detach().item(), d_logits_src.detach().item())

            #print(domain_perf)
            #print(epoch_domain_loss_src/counter)
            #exit(1)

            

            domain_acc = float(domain_correct / domain_total) if domain_total > 0 else 0.0

            if save_every_epoch:
                epoch_save_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                torch.save({
                    "total_epochs": self.maxiter,
                    "epoch": epoch,
                    "model_state_dict": epoch_save_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hidden_layer_size": self.hidden_layer_size,
                    "hidden_dims": self.hidden_dims,
                }, os.path.join(main_dir,"network_ckpt_"+str(epoch)+".pt"))
            
            #save training risk
            if True:
                self.model.eval()
                with torch.no_grad():
                    preds = self.model.predict(Xs_t).cpu().numpy()
                train_risk = np.mean(preds != self._numpy_labels(Y))
                training_risk_history.append(train_risk)

            writer.add_scalar('Train risk (epoch)', train_risk, epoch)
            writer.add_scalar('Label Loss (Epoch)', epoch_label_loss/counter, epoch)
            writer.add_scalar('Domain loss src (Epoch)', epoch_domain_loss_src/counter, epoch)
            writer.add_scalar('domain loss tgt (Epoch)', epoch_domain_loss_tgt/counter, epoch)
            writer.add_scalar('Domain Accuracy (Epoch)', domain_acc, epoch)
            #writer.add_scalar('Domain acc: ', 
            #                  (domain_perf['src'][0] + domain_perf['tgt'][0])/(domain_perf['src'][0] + domain_perf['tgt'][0] + domain_perf['src'][1] + domain_perf['tgt'][1]),
            #                  epoch)
            # ---- Early stopping on validation set ----
            if X_valid is not None:
                self.model.eval()
                with torch.no_grad():
                    preds = self.model.predict(Xv_t).cpu().numpy()
                valid_risk = np.mean(preds != self._numpy_labels(Y_valid))
                val_risk_history.append(valid_risk)
                writer.add_scalar('valid_risk (Epoch)', valid_risk, epoch)
                if valid_risk <= best_valid_risk:
                    best_valid_risk = valid_risk
                    best_weights    = {k: v.clone() for k, v in self.model.state_dict().items()}
                    best_epoch      = epoch
                    continue_until  = max(continue_until, int(1.5 * epoch))
                elif early_valid_stop and epoch > continue_until:
                    if self.verbose:
                        print(f'[DANN early stop] epoch {epoch}')
                    break
            
            if X_test is not None:
                self.model.eval()
                with torch.no_grad():
                    preds = self.model.predict(Xt_t).cpu().numpy()
                test_risk = np.mean(preds != self._numpy_labels(Y_test))
                test_risk_history.append(test_risk)
                writer.add_scalar('test_risk (Epoch)', test_risk, epoch)


        # Restore best weights if early stopping was used
        if False and not save_every_epoch:
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_weights,
            }, os.path.join(main_dir,"run_id:"+str(run_id)+".pt"))


        if X_valid is not None and best_weights is not None:
            self.model.load_state_dict(best_weights)
            self.nb_iter    = best_epoch
            self.valid_risk = best_valid_risk
        else:
            self.nb_iter    = self.maxiter
            self.valid_risk = 2.0
        
        return training_risk_history,val_risk_history

    def predict(self, X) -> np.ndarray:
        """Return predicted class labels for X."""
        self.model.eval()
        with torch.no_grad():
            X_t = self._to_tensor(X)
            return self.model.predict(X_t).cpu().numpy()

    def predict_domain(self, X) -> np.ndarray:
        """Return predicted domain (0=target, 1=source) for X."""
        self.model.eval()
        with torch.no_grad():
            X_t  = self._to_tensor(X)
            h    = self.model.feature_extractor(X_t)
            # Call the linear layer directly, bypassing the GRL
            logits = self.model.domain_discriminator.layer(h)
            probs  = torch.sigmoid(logits).squeeze(1)
            return (probs >= 0.5).long().cpu().numpy()

    def forward(self, X) -> np.ndarray:
        """
        Return softmax class probabilities for X.
        Shape: (num_classes, n_examples) — matches original API.
        """
        self.model.eval()
        with torch.no_grad():
            X_t      = self._to_tensor(X)
            h        = self.model.feature_extractor(X_t)
            logits   = self.model.label_predictor(h)
            probs    = torch.softmax(logits, dim=1)
            return probs.T.cpu().numpy()

    def hidden_representation(self, X) -> np.ndarray:
        """Return hidden layer activations for X. Shape: (n_examples, hidden_size)."""
        self.model.eval()
        with torch.no_grad():
            X_t = self._to_tensor(X)
            return self.model.hidden_representation(X_t).cpu().numpy()