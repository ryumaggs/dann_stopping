from sklearn.datasets import load_svmlight_files
import numpy as np
import os
from PIL import Image
from sklearn.datasets import fetch_openml

MNIST_CACHE_PATH = './results_and_data/data/mnist_784.npz'

def _load_mnist_cached():
    """Load MNIST from local cache if available, otherwise fetch and save."""
    if os.path.exists(MNIST_CACHE_PATH):
        cache = np.load(MNIST_CACHE_PATH)
        return cache['data'], cache['target']
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    data   = mnist.data.astype(np.float32) / 255.0
    target = mnist.target.astype(int)
    os.makedirs(os.path.dirname(MNIST_CACHE_PATH), exist_ok=True)
    np.savez(MNIST_CACHE_PATH, data=data, target=target)
    return data, target


def load_amazon(source_name, target_name, data_folder=None, verbose=False):
    """
    Load the amazon sentiment datasets from svmlight format files
    inputs:
        source_name : name of the source dataset
        target_name : name of the target dataset
        data_folder : path to the folder containing the files
    outputs:
        xs : training source data matrix
        ys : training source label vector
        xt : training target data matrix
        yt : training target label vector
        xtest : testing target data matrix
        ytest : testing target label vector
    """

    if data_folder is None:
        data_folder = 'results_and_data/data/'

    source_file = data_folder + source_name + '_train.svmlight'
    target_file = data_folder + target_name + '_train.svmlight'
    test_file = data_folder + target_name + '_test.svmlight'

    if verbose:
        print('source file:', source_file)
        print('target file:', target_file)
        print('test file:  ', test_file)

    xs, ys, xt, yt, xtest, ytest = load_svmlight_files([source_file, target_file, test_file])

    # Convert sparse matrices to numpy 2D array
    xs, xt, xtest = (np.array(X.todense()) for X in (xs, xt, xtest))

    # Convert {-1,1} labels to {0,1} labels
    ys, yt, ytest = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, ytest))

    return xs, ys, xt, yt, xtest, ytest



def load_mnist_to_mnistm(MNIST_m_path, K_source=None, K_target=None, verbose=False):
    """
    Load MNIST as source and MNIST-M as target, matching the interface of load_amazon.
    MNIST grayscale images are resized to 32x32 and replicated to 3 channels
    to match MNIST-M's RGB format. Both datasets have feature dimension 3072.

    inputs:
        MNIST_m_path : path to the MNIST-M root directory, which should contain:
                       mnist_m_train/            (folder of .png images)
                       mnist_m_test/             (folder of .png images)
                       mnist_m_train_labels.txt
                       mnist_m_test_labels.txt
        K_source     : number of MNIST source samples to load (default: all 60000)
        K_target     : number of MNIST-M target train samples to load (default: all)
    outputs:
        xs    : source training data matrix    (MNIST,   shape [K_source, 3, 32, 32])
        ys    : source training label vector   (MNIST,   shape [K_source])
        xt    : target training data matrix    (MNIST-M, shape [K_target, 3, 32, 32])
        yt    : target training label vector   (MNIST-M, shape [K_target])
        xtest : target test data matrix        (MNIST-M test, shape [N_test, 3, 32, 32])
        ytest : target test label vector       (MNIST-M test, shape [N_test])
    """

    # ── Helper: convert a flat 784-dim MNIST image to 32x32 RGB ──────────────
    def mnist_to_rgb(flat_image):
        img = Image.fromarray(
            (flat_image.reshape(28, 28) * 255).astype(np.uint8)
        )
        img = img.resize((32, 32), Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0   # [32, 32]
        return np.stack([img_array, img_array, img_array], axis=0)  # [3, 32, 32]

    # ── Load MNIST as source ──────────────────────────────────────────────────
    if verbose:
        print('Loading MNIST source dataset...')

    xs_raw, ys = _load_mnist_cached()

    # Use standard 60k training split, shuffle, then apply K_source if specified
    xs_raw = xs_raw[:60000]
    ys     = ys[:60000]

    perm   = np.random.permutation(len(ys))
    xs_raw = xs_raw[perm]
    ys     = ys[perm]

    if K_source is not None:
        xs_raw = xs_raw[:K_source]
        ys     = ys[:K_source]

    if verbose:
        print(f'Converting {xs_raw.shape[0]} MNIST images to 32x32 RGB...')

    xs = np.array([mnist_to_rgb(x) for x in xs_raw])   # [K_source, 3072]

    if verbose:
        print(f'MNIST source loaded: {xs.shape[0]} samples, {xs.shape[1]} features')

    # ── Helper: load an MNIST-M split from folder + label file ───────────────
    def load_mnistm_split(image_folder, label_file, K=None):
        if verbose:
            print(f'Loading MNIST-M from {image_folder}')
            print(f'Label file: {label_file}')

        with open(label_file, 'r') as f:
            entries = [line.strip().split() for line in f if line.strip()]

        np.random.shuffle(entries)

        if K is not None:
            entries = entries[:K]

        images = []
        labels = []

        for filename, label in entries:
            img_path = os.path.join(image_folder, filename)
            img      = Image.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0   # [32, 32, 3]
            images.append(img_array.transpose(2, 0, 1))            # [3, 32, 32]
            labels.append(int(label))

        X = np.array(images)          # [K, 3, 32, 32]
        y = np.array(labels, dtype=int)

        if verbose:
            print(f'Loaded {X.shape[0]} samples, {X.shape[1]} features')

        return X, y

    # ── Load MNIST-M train as target ─────────────────────────────────────────
    train_image_folder = os.path.join(MNIST_m_path, 'mnist_m_train')
    train_label_file   = os.path.join(MNIST_m_path, 'mnist_m_train_labels.txt')
    xt, yt = load_mnistm_split(train_image_folder, train_label_file, K=K_target)

    # ── Load MNIST-M test (always load all test samples) ─────────────────────
    test_image_folder = os.path.join(MNIST_m_path, 'mnist_m_test')
    test_label_file   = os.path.join(MNIST_m_path, 'mnist_m_test_labels.txt')
    xtest, ytest = load_mnistm_split(test_image_folder, test_label_file)

    return xs, ys, xt, yt, xtest, ytest


def load_mnistm_to_mnist(MNIST_m_path, K_source=None, K_target=None, verbose=False):
    """
    Load MNIST-M as source and MNIST as target.
    MNIST-M RGB images are converted to grayscale via luminance weighting,
    then replicated to 3 channels. MNIST grayscale images are replicated to
    3 channels. Both datasets output [3, 32, 32] to match the model's Conv2d(3,...).

    inputs:
        MNIST_m_path : path to the MNIST-M root directory, which should contain:
                       mnist_m_train/            (folder of .png images)
                       mnist_m_test/             (folder of .png images)
                       mnist_m_train_labels.txt
                       mnist_m_test_labels.txt
        K_source     : number of MNIST-M source samples to load (default: all)
        K_target     : number of MNIST target train samples to load (default: all 60000)
    outputs:
        xs    : source training data matrix    (MNIST-M, shape [K_source, 3, 32, 32])
        ys    : source training label vector   (MNIST-M, shape [K_source])
        xt    : target training data matrix    (MNIST,   shape [K_target, 3, 32, 32])
        yt    : target training label vector   (MNIST,   shape [K_target])
        xtest : target test data matrix        (MNIST test, shape [N_test, 3, 32, 32])
        ytest : target test label vector       (MNIST test, shape [N_test])
    """

    # ── Helper: load an MNIST-M split, downproject RGB -> grayscale, replicate to 3ch ──
    def load_mnistm_split_gray(image_folder, label_file, K=None):
        if verbose:
            print(f'Loading MNIST-M from {image_folder}')
            print(f'Label file: {label_file}')

        with open(label_file, 'r') as f:
            entries = [line.strip().split() for line in f if line.strip()]

        np.random.shuffle(entries)

        if K is not None:
            entries = entries[:K]

        images = []
        labels = []

        for filename, label in entries:
            img_path  = os.path.join(image_folder, filename)
            img       = Image.open(img_path).convert('RGB')
            img_gray  = img.convert('L')                                        # luminance-weighted grayscale
            img_gray  = img_gray.resize((32, 32), Image.BILINEAR)
            img_array = np.array(img_gray, dtype=np.float32) / 255.0           # [32, 32]
            images.append(np.stack([img_array, img_array, img_array], axis=0)) # [3, 32, 32]
            labels.append(int(label))

        X = np.array(images)          # [K, 3, 32, 32]
        y = np.array(labels, dtype=int)

        if verbose:
            print(f'Loaded {X.shape[0]} samples')

        return X, y

    # ── Load MNIST-M train as source ──────────────────────────────────────────
    train_image_folder = os.path.join(MNIST_m_path, 'mnist_m_train')
    train_label_file   = os.path.join(MNIST_m_path, 'mnist_m_train_labels.txt')
    xs, ys = load_mnistm_split_gray(train_image_folder, train_label_file, K=K_source)

    if verbose:
        print(f'MNIST-M source loaded: {xs.shape[0]} samples')

    # ── Helper: convert a flat 784-dim MNIST image to 32x32, replicate to 3ch ─
    def mnist_to_3ch(flat_image):
        img = Image.fromarray(
            (flat_image.reshape(28, 28) * 255).astype(np.uint8)
        )
        img = img.resize((32, 32), Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0   # [32, 32]
        return np.stack([img_array, img_array, img_array], axis=0)  # [3, 32, 32]

    # ── Load MNIST as target ──────────────────────────────────────────────────
    if verbose:
        print('Loading MNIST target dataset...')

    xt_raw, yt_all = _load_mnist_cached()

    # Standard 60k train / 10k test split
    xt_train_raw = xt_raw[:60000]
    yt_train     = yt_all[:60000]
    xt_test_raw  = xt_raw[60000:]
    yt_test      = yt_all[60000:]

    # Shuffle train split then apply K_target
    perm         = np.random.permutation(len(yt_train))
    xt_train_raw = xt_train_raw[perm]
    yt_train     = yt_train[perm]

    if K_target is not None:
        xt_train_raw = xt_train_raw[:K_target]
        yt_train     = yt_train[:K_target]

    xt    = np.array([mnist_to_3ch(x) for x in xt_train_raw])  # [K_target, 3, 32, 32]
    xtest = np.array([mnist_to_3ch(x) for x in xt_test_raw])   # [N_test,   3, 32, 32]

    if verbose:
        print(f'MNIST target loaded: {xt.shape[0]} train, {xtest.shape[0]} test samples')

    return xs, ys, xt, yt_train, xtest, yt_test