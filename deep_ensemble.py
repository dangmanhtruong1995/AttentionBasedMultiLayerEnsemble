import copy
import pdb
import numpy as np
import time
from os.path import join as pjoin
from pdb import set_trace
from sklearn.metrics import accuracy_score
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pylab import plt

# Import the optimization libraries
from bga import BGA
import pyswarms as ps
# from scipy.optimize import differential_evolution
import scipy
from differential_evolution import differential_evolution

# Import the custom libraries
from base_layer import BaseLayer
from encoding_helper import BaseEncodingWrapper, LayerEncodingWrapper
from ensemble_layer import EnsembleLayer
from utils import collect_result, roulette_selection
from attention_based_ensemble import run_transformer_decoder

   
def plot_training_and_validation_loss(train_loss_list, val_loss_list, result_path):
    epochs = range(1, len(train_loss_list)+1) 
    # Plot and label the training and validation loss values
    plt.plot(epochs, train_loss_list, label='Training Loss')
    plt.plot(epochs, val_loss_list, label='Validation Loss')
     
    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
     
    # Set the tick locations
    plt.xticks(np.arange(0, len(train_loss_list)+1, 2))
     
    # Display the plot
    plt.legend(loc='best')
    plt.show()
    # plt.savefig(pjoin("result", self.dataset_name, "train_val_loss_layer_%d.png" % (layer_idx)))
    plt.savefig(result_path)
    plt.close()
        
class DeepEnsemble:
    """ Class for the deep ensemble evolutionary connection model
    """

    def __init__(self, config, file_name):
        self.config = config
        self.meta_clf = config['meta_clf']
        self.clfs = config['clfs']
        self.n_cv_folds = config['n_cv_folds']
        self.n_cls = config['n_cls']
        self.n_clfs = config['n_clfs']
        self.early_stopping_rounds = config['early_stopping_rounds']
        self.max_layers = config['max_layers']
        self.dataset_name = file_name
        
    def optimize(self, X_train, y_train, X_val, y_val):
        """ Find the optimal encoding for each layer

        Parameters
        ----------
        X_train: Train data, of size (n_train_inst, n_features)

        y_train: Train labels, of size (n_train_inst)

        X_val: Validation data, of size (n_val_inst, n_features)

        y_val: Validation labels, of size (n_val_inst)

        Returns
        -------
        best_n_layers: Optimal number of layers

        enc_list: List of encodings for each layer

        acc_list: List of accuracy for each layer

        """

        config = self.config
        meta_clf = self.meta_clf
        clfs = self.clfs
        n_cv_folds = self.n_cv_folds
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        n_features = X_train.shape[1]
        early_stopping_rounds = self.early_stopping_rounds
        max_layers = self.max_layers
        
        enc_list = []
        acc_list = []
        layer_idx = 0
        n_layers = 0
        best_acc = -1
        best_n_layers = -1
        X_train_in = copy.deepcopy(X_train)
        X_val_in = copy.deepcopy(X_val)

        print("Start searching...")

        while True:
            # Find the optimal encoding
            print('------------------------')
            print('Optimizing layer {}'.format(layer_idx))
            if layer_idx == 0:
                # First, precompute metadata
                first_layer = BaseLayer(
                    copy.deepcopy(clfs),
                    n_cv_folds,
                    n_cls,
                    n_clfs,
                    meta_clf)
                enc_all = np.ones(n_clfs, dtype=np.uint8)
                (precompute_metadata_train, precompute_metadata_val, pred_val) = \
                    first_layer.run(X_train_in, y_train, X_val_in, y_val, enc_all)

                # Then, optimize
                first_layer = BaseLayer(
                    copy.deepcopy(clfs),
                    n_cv_folds,
                    n_cls,
                    n_clfs,
                    meta_clf,
                    precompute_metadata_train=precompute_metadata_train,
                    precompute_metadata_val=precompute_metadata_val,
                    )
                n_dim_current_layer = n_clfs
                
                t1 = time.time() 
                enc = enc_all
                acc = accuracy_score(y_val, pred_val)
                # set_trace()
                t2 = time.time()                
                print('Running current layer took: %f seconds.' % (t2-t1))
                
                if int(np.sum(enc)) == 0:
                    enc[enc == 0] = 1
                enc_wrapper = BaseEncodingWrapper(enc, n_clfs)
                # set_trace()
                (X_train_in, X_val_in, _) = first_layer.run(
                    X_train_in, y_train, X_val_in, y_val, enc)
                n_clfs_prev = enc_wrapper.num_of_clfs_used()
            else:
                # set_trace()
            
                current_layer = EnsembleLayer(
                    copy.deepcopy(clfs),
                    n_cv_folds,
                    n_cls,
                    n_clfs,
                    n_clfs_prev,
                    meta_clf)
                n_dim_current_layer = n_clfs * (n_clfs_prev + 1)
                
                t1 = time.time()                
                
                # First, take all the classifiers and run through the transformer decoder
                enc = np.ones((n_dim_current_layer))
                enc = enc.astype(np.int32)
                enc_wrapper = LayerEncodingWrapper(enc,
                    n_clfs_prev, n_clfs)
                (X_train_curr, X_val_curr, _) = current_layer.run(
                    X_train_in, y_train, X_val_in, y_val, X_train, X_val, enc)

                np.savetxt(
                    pjoin("result", self.dataset_name, "metadata_prev_layer_%d.txt" % (layer_idx)),
                    X_train_in, fmt="%.6f", delimiter=",",
                )
                np.savetxt(
                    pjoin("result", self.dataset_name, "metadata_layer_%d.txt" % (layer_idx)),
                    X_train_curr, fmt="%.6f", delimiter=",",
                )
                np.savetxt(
                    pjoin("result", self.dataset_name, "gt_layer_%d.txt" % (layer_idx)),
                    y_train, fmt="%.6f", delimiter=",",
                )
                
                attn_score, train_loss_list, val_loss_list = run_transformer_decoder(X_train_in, X_train_curr, y_train,
                    n_cls, n_clfs_prev, n_clfs, n_epoch=config['attention_n_epoch'], batch_size=config['attention_batch_size'])
                
                plot_result_path = pjoin("result", self.dataset_name, "train_val_loss_layer_%d.png" % (layer_idx))
                plot_training_and_validation_loss(train_loss_list, val_loss_list, plot_result_path)
                
                attn_score_all = np.sum(attn_score, axis=0) # (n_clfs, n_clfs_prev)
                attn_score_all = np.transpose(attn_score_all) # (n_clfs_prev, n_clfs)
                np.savetxt(
                    pjoin("result", self.dataset_name, "attention_unnormalized_layer_%d.txt" % (layer_idx)),
                    attn_score_all,
                    fmt="%.6f",
                    delimiter=",",
                )                
                attn_score_all_normalized = attn_score_all / np.sum(attn_score_all, axis=1)[:, np.newaxis]
                # attn_score_all_normalized = attn_score_all / np.sum(attn_score_all, axis=0)[:, np.newaxis]
                                
                np.savetxt(
                    pjoin("result", self.dataset_name, "attention_normalized_layer_%d.txt" % (layer_idx)),
                    attn_score_all_normalized,
                    fmt="%.6f",
                    delimiter=",",
                )
                
                # prob = 1-prob
                enc = np.zeros((n_clfs_prev+1, n_clfs))
                
                # Keep a percentage of classifiers with the highest attention scores (for each column)
                n_keep_attn = int(n_clfs*config['attention_keep_percent'])
                print("n_keep_attn: %d" % (n_keep_attn))
                for idx_col in range(n_clfs):
                    attn_col = attn_score_all_normalized[:, idx_col]
                    idx_sort_descend = np.argsort(attn_col)[::-1]
                    idx_chosen_list = idx_sort_descend[:n_keep_attn]
                    for idx_chosen in idx_chosen_list:
                        enc[idx_chosen+1, idx_col] = 1 # The first row is for the original data
                    # set_trace()
                        
                enc = enc.astype(np.int32)
                enc[0, :] = 1 # Always get the original data 
                enc = np.transpose(enc) # LayerEncodingWrapper expects the encoding to be of size (n_clfs, (n_clfs_prev+1))
                np.savetxt(
                    pjoin("result", self.dataset_name, "encoding_layer_%d.txt" % (layer_idx)),
                    enc,
                    fmt="%d",
                    delimiter=" ",
                )  
                enc = enc.flatten()
                # set_trace()
                # if int(np.sum(enc)) == 0:
                    # enc[enc == 0] = 1
                enc_wrapper = LayerEncodingWrapper(enc,
                    n_clfs_prev, n_clfs)
                (X_train_in, X_val_in, pred_val) = current_layer.run(
                    X_train_in, y_train, X_val_in, y_val, X_train, X_val, enc)  
                acc = accuracy_score(y_val, pred_val)
                # if int(np.sum(enc)) == 0:
                    # enc[enc == 0] = 1
                n_clfs_prev = enc_wrapper.num_of_clfs_out()
                
                t2 = time.time()
                print('Running current layer took: %f seconds.' % (t2-t1))

            acc_list.append(acc)
            enc_list.append(enc)
            print("Finish optimizing for current layer!")
            enc_wrapper.print_info()
            print("Fitness:")
            print(acc)
            if best_acc < acc:
                best_acc = acc
                best_n_layers = layer_idx
            else:
                if layer_idx - best_n_layers >= early_stopping_rounds:
                    break
            layer_idx += 1
            if (best_n_layers+1) > max_layers:
                break

        if best_n_layers == 0:
            best_n_layers = 1 # At least 2 layers
           
        
        return (best_n_layers, enc_list, acc_list)

    def fit_and_test(self, X_trainval, y_trainval, X_test, y_test,
                     n_layers, enc_list):
        """ Fit and test in one function so that we don't have to
            store all intermediate layers

        Parameters
        ----------
        X_trainval: Trainval data, of size (n_trainval_inst, n_features)

        y_trainval: Trainval labels, of size (n_trainval_inst)

        X_test: Test data, of size (n_test_inst, n_features)

        y_test: Test labels, of size (n_test_inst)

        n_layers: Number of optimal layers

        enc_list: List of encoding (list of list)

        Returns
        -------
        metric_dict: A dictionary containing metrics for test result
        on each layer
        """

        meta_clf = self.meta_clf
        clfs = self.clfs
        n_cv_folds = self.n_cv_folds
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        n_features = X_trainval.shape[1]

        metric_dict = {}
        metric_dict['acc'] = []
        metric_dict['precision'] = []
        metric_dict['recall'] = []
        metric_dict['f1'] = []

        layer_idx = 1
        # n_layers = 0
        best_acc = -1
        best_n_layers = -1

        X_trainval_in = copy.deepcopy(X_trainval)
        X_test_in = copy.deepcopy(X_test)

        print("Test after optimize!")
        X_test_in_list = []
        n_clfs_prev_list = []
        # for layer_idx in range(len(enc_list)):
        for layer_idx in range(n_layers+1):
            enc = enc_list[layer_idx]
            if isinstance(enc, list):
                enc = np.array(enc, dtype=np.int32)
            if layer_idx == 0:
                # First layer
                first_layer = BaseLayer(clfs, n_cv_folds, n_cls, n_clfs, meta_clf)
                (X_trainval_in, X_test_in, pred_test) = first_layer.run(
                    X_trainval_in, y_trainval, X_test_in, y_test, enc)
                enc_wrapper = BaseEncodingWrapper(enc, n_clfs)
                n_clfs_prev = enc_wrapper.num_of_clfs_used()
            else:
                # From 2nd layer onward
                current_layer = EnsembleLayer(clfs, n_cv_folds, n_cls,
                                              n_clfs, n_clfs_prev, meta_clf)
                (X_trainval_in, X_test_in, pred_test) = current_layer.run(
                    X_trainval_in, y_trainval, X_test_in, y_test, X_trainval, X_test, enc)

                enc_wrapper = LayerEncodingWrapper(enc,
                                                   n_clfs_prev, n_clfs)
                n_clfs_prev = enc_wrapper.num_of_clfs_out()
            
            n_clfs_prev_list.append(n_clfs_prev)
            
            result = collect_result(y_test, pred_test)
            for metric in metric_dict:
                metric_dict[metric].append(result[metric])
            acc = result['acc']

            print("")
            print("Layer: %d" % (layer_idx))
            enc_wrapper.print_info()
            print("Fitness:")
            print(acc)

        return metric_dict
