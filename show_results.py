import numpy as np
import os
import matplotlib.pyplot as plt
from cvae import ClassificationVariationalNetwork
from data import torch_load as torchdl
from utils.save_load import collect_networks, get_path_from_input
from roc_curves import ood_roc, miss_roc, fpr_at_tpr, tpr_at_fpr
import argparse
import torch
from data.torch_load import choose_device

def showable_tensor(x):

    return x.squeeze().cpu().detach().numpy()


def show_y_matrix(vae, x):

    x_, y_, loss = vae.evaluate(x)
    logits = (y_ / (1 - y_)).log()

    plt.imshow(logits.cpu().detach().numpy())
    plt.show()
    pass


def show_x(vae, x):

    x_, y_, oneloss = vae.evaluate(x)
    
    for i, x in enumerate(x_):

        plt.figure()
        plt.imshow(showable_tensor(x))
        # plt.show()
        plt.title(f'y={i} loss = {oneloss[i]}')

    plt.show()

    pass


def show_x_y(vae, x, title='', predict_method='mean'):
    """ x is ONE sample 
    """
    
    x_reco, y_est_, oneloss = vae.evaluate(x)
    y_pred_ = y_est_.mean(0).cpu().numpy()

    y_pred = vae.predict(x, method=predict_method)

    y_ = torch.cat([y_est_.cpu(),
                    y_est_.mean(0).cpu().unsqueeze(0)], 0)
    # print(y_)

    log_px = vae.log_px(x)
    print(f'{title}{" :" if title else ""} log_px = {log_px}\n')
    
    f, axes = plt.subplots(3, 4)

    axes = axes.reshape(12)
    
    axes[0].imshow(showable_tensor(x), cmap='gray')
    axes[0].set_title(f'original ({title}) log_px={log_px:.3e}')

    ax_i = 1
    for i, x in enumerate(x_reco):

        axes[ax_i].imshow(showable_tensor(x), cmap='gray')
        # plt.show()
        warn = '[]' if i == y_pred else ('', '')
        axes[ax_i].set_title(f'{warn[0]}y={i} loss = {oneloss[i]:.3e}{warn[-1]}')
        ax_i += 1
        
    logits = (y_ / (1 - y_)).log()

    axes[-1].imshow(showable_tensor(logits), cmap='gray')

    axes[-1].set(xlabel='p(y) output', ylabel='y input')
    axes[-1].set_title(f'y_pred={np.round(100 * y_pred_)}')

    return f

                
def show_examples(vae, testset, oodset, num_of_examples=10,
                  batch_size=100, num_batch=5, export=False,
                  export_dir='/tmp', device=None):

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=True)

    oodloader = torch.utils.data.DataLoader(oodset,
                                             batch_size=batch_size,
                                             shuffle=True)
    device = choose_device(device)
    print(f'...{device}...')

    example = 0
    continued = True
    while example < num_of_examples and continued:
        
        data = next(iter(testloader))
        x_test = data[0].to(device)
        y_test = data[1].to(device)

        data = next(iter(oodloader))
        x_ood = data[0].to(device)
        y_ood = data[1].to(device)

        y_pred = dict()
        idx_pred = dict()
        idx_miss = dict()
        acc = dict()

        for m in vae.predict_methods:
            print(x_test.device)
            y_pred[m] = vae.predict(x_test, method=m)

            idx_pred[m] = torch.where(y_test == y_pred[m])[0]
            idx_miss[m] = torch.where(y_test != y_pred[m])[0]

            acc[m] = len(idx_pred[m]) / len(x_test)

            print(f'accurary ({m}): {acc[m]}')

            i_pred = idx_pred[m][np.random.randint(0, len(idx_pred))]
            i_miss = idx_miss[m][np.random.randint(0, len(idx_miss))]
            i_ood = np.random.randint(len(x_ood))
        
            f0 = show_x_y(vae, x_test[i_pred], title=f'y_true={y_test[i_pred]} by {m}')
            f0.show()
    
            f1 = show_x_y(vae, x_ood[i_ood], title=f'ood')
            f1.show()

            x_miss, y_miss = x_test[i_miss], y_test[i_miss]
            f2 = show_x_y(vae, x_miss, title=f'y_missed={y_miss}')
            f2.show()

        f, a = plt.subplots(1, 1)
        fpr, tpr = ood_roc(vae, testset, oodset,
                           batch_size=batch_size,
                           num_batch=num_batch)
        fpr_at_tpr95 = fpr_at_tpr(fpr, tpr, 0.95)
        fpr_at_tpr98 = fpr_at_tpr(fpr, tpr, 0.99)
        a.plot(100 * fpr, 100 * tpr)
        a.plot(100 * fpr, 100 * fpr, 'r--')
        a.set_xlabel('false positive of ood')
        a.set_ylabel('true positive of ood')
        a.set_title('ood roc fpr at tpr 95%-98%: ' +
                    f'{100*fpr_at_tpr95:.1f}-{100*fpr_at_tpr98:.1f}')

        f.show()

        f, axis = plt.subplots(1, len(vae.predict_methods))
        for m, a in zip(vae.predict_methods, axis):
            fpr, tpr = miss_roc(vae, testset,
                                predict_method=m,
                                batch_size=batch_size,
                                num_batch=num_batch)

            fpr_at_tpr95 = fpr_at_tpr(fpr, tpr, 0.95)
            fpr_at_tpr98 = fpr_at_tpr(fpr, tpr, 0.99)
            a.plot(100 * fpr, 100 * tpr)
            a.plot(100 * fpr, 100 * fpr, 'r--')
            a.set_xlabel('false positive of miss')
            a.set_ylabel('true positive of miss')
            a.set_title(f'miss roc by {m}, fpr at tpr 95%-98%: ' +
                        f'{100*fpr_at_tpr95:.1f}-{100*fpr_at_tpr98:.1f}')

        f.show()
        
        char = input()
        if char != '':
            plt.close('all')
        if char.lower() == 'q':
            continued = False

            
def plot_accuracies(list_of_vae, ax, method, semilog=True, verbose=0):

    beta_ = []
    acc_ = []

    for vae_dict in list_of_vae:
        acc = vae_dict['acc'][method]
        if acc is not None:
            beta_.append(vae_dict['net'].beta)
            acc_.append(acc)

    beta_sorted = np.sort(beta_)
    i = np.argsort(beta_)
    acc_sorted = [acc_[_] for _ in i]

    acc_max = np.max(acc_)
    beta_max = beta_[np.argmax(acc_)]

    if verbose > 0:
        dir_of_vae = vae_dict['dir']
        print(f'Prediction error with {method} for {dir_of_vae}')
    if verbose > 1:
        for (b, a) in zip(beta_sorted, acc_sorted):
            print(f'{b:.2e}: {100-100*a:4.1f} %')

    legend = vae_dict['net'].print_architecture()
    if semilog:
        ax.semilogx(beta_sorted, [1 - _ for _ in acc_sorted], '.', label=legend)
    else:
        ax.plot(beta_sorted, [1 - _ for _ in acc_sorted], '.', label=legend)

    return beta_sorted, acc_sorted
    

def plot_fpr(list_of_vae, ax, tpr, semilog=True, verbose=0):

    beta_ = []
    fpr_ = []

    for vae_dict in list_of_vae:
        fpr = vae_dict['fpr at tpr'][tpr]
        if fpr is not None:
            beta_.append(vae_dict['net'].beta)
            fpr_.append(fpr)

    beta_sorted = np.sort(beta_)
    i = np.argsort(beta_)
    fpr_sorted = [fpr_[_] for _ in i]

    fpr_min = np.min(fpr_)
    beta_min = beta_[np.argmin(fpr_)]

    if verbose > 0:
        dir_of_vae = vae_dict['dir']
        print(f'FPR with TPR={tpr} for {dir_of_vae}')
    if verbose > 1:
        for (b, r) in zip(beta_sorted, fpr_sorted):
            print(f'{b:.2e}: {100-100*r:4.1f} %')

    legend = vae_dict['net'].print_architecture()
    if semilog:
        ax.semilogx(beta_sorted, fpr_sorted, '.', label=legend)
    else:
        ax.plot(beta_sorted, fpr_sorted, '.', label=legend)

    return beta_sorted, fpr_sorted
    

if __name__ == '__main__':

    description="show results of networks in directory"
    parser = argparse.ArgumentParser(description=description)
        
    parser.add_argument('--dataset', default='fashion',
                        choices=['fashion', 'mnist'])

    parser.add_argument('directories',
                        help='where to find the networks',
                        nargs='*', default=None)
    
    parser.add_argument('-N', '--num-batch', default=5, type=int)
    parser.add_argument('-b', '--batch-size', default=100, type=int)
    parser.add_argument('-x', '--export', action='store_true')

    parser.add_argument('-d', '--export-dir', default=None,
                        help='directory to export results')

    parser.add_argument('-c', '--force-cpu', action='store_true')

    parser.add_argument('-v', '--verbose', action='count', default=0)
    
    args = parser.parse_args()

    verbose = args.verbose
    export_dir = args.export_dir
    export = args.export or export_dir is not None
    batch_size = args.batch_size
    num_batch = args.num_batch
    dataset = args.dataset
    directories = args.directories
    # print(directories)

    force_cpu = args.force_cpu
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if force_cpu or not has_cuda else 'cuda')
    print(f'Used device: {device}') # ({force_cpu}, {has_cuda})')
    
    if len(directories) == 0:
        jobs = os.path.join(os.getcwd(), 'jobs')
        if not os.path.exists(jobs):
            jobs = os.getcwd()
        directories = [get_path_from_input(jobs)]

    if verbose > 0:
        print(f'Used device: {device}')
        print(f'Explored director{"ies" if len(directories) > 1 else "y"}:')
        [print(d) for d in directories]
        print(f'with assumed dataset {dataset}')

    tpr_ = [95, 98]
                    
    if dataset == 'fashion':

        trainset, testset = torchdl.get_fashion_mnist()
        _, oodset = torchdl.get_mnist()
        oodset.name = 'mnist'

    if dataset == 'mnist':
        trainset, testset = torchdl.get_mnist()

    list_of_lists_of_vae = []
    
    with torch.no_grad():
        for directory in directories:
            collect_networks(directory, list_of_lists_of_vae,
                             testset=testset, oodset=oodset,
                             true_pos_rates=tpr_,
                             batch_size=batch_size, num_batch=num_batch,
                             device=device, verbose=verbose)
    
    num_of_nets = sum([len(_) for _ in list_of_lists_of_vae])

    if num_of_nets == 0:
        print('NOTHING TO SEE HERE\n')

    if num_of_nets == 1:
        # print(f'...{device}...')
        vae_dict = list_of_lists_of_vae[0][0]
        vae = vae_dict['net']
        directory = vae_dict['dir']
        with torch.no_grad():
            # print(f'***{device}***')
            show_examples(vae, testset, oodset, batch_size=batch_size,
                          num_batch=num_batch, device=device)

    if num_of_nets > 1:
        dict_of_acc_f = dict()
        dict_of_acc_axes = dict()

        f_fpr, a_fpr = plt.subplots(1, len(tpr_))

        methods = ClassificationVariationalNetwork.predict_methods
        for m in methods:
            dict_of_acc_f[m], dict_of_acc_axes[m] = plt.subplots()
        
        for list_of_nets in list_of_lists_of_vae:

            for a, rate in enumerate(tpr_):
                beta_, fpr_ = plot_fpr(list_of_nets,
                                       a_fpr[a],
                                       rate,
                                       verbose=verbose)

            for m in methods:
                beta_, acc_ = plot_accuracies(list_of_nets,
                                              dict_of_acc_axes[m], m,
                                              verbose=verbose)
                
                if export_dir is not None:
                    arch = list_of_nets[0]['net'].print_architecture()
                    file_path = os.path.join(export_dir, arch + f'_{m}.tab')
                    with open(file_path, 'w+') as f:
                        for b, a in zip(beta_, acc_):
                            f.write(f'{b:.2e} {a:6.4f}\n')
        for m in methods:
            dict_of_acc_axes[m].legend(loc='upper right')
            dict_of_acc_axes[m].set_title(f'Accuracy with predict method {m}')
            dict_of_acc_f[m].show()

        for a, rate in enumerate(tpr_):
            a_fpr[a].legend(loc='upper right')
            a_fpr[a].set_title(f'FPR at TPR={rate}%')
            f_fpr.show()

        input()
