import math
# import networkx as nx
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)


options = {
    'node_size': 2,
    'edge_color' : 'black',
    'linewidths': 1,
    'width': 0.5
}

def plot_graphs_list(graphs, title='title', max_num=16, save_dir=None, N=0):
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()

    for i in range(max_num):
        # idx = i * (batch_size // max_num)
        idx = i + max_num*N
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)

        ax = plt.subplot(img_c, img_c, i + 1)
        # title_str = f'e={e - l}, n={v}'
        # if 'lobster' in save_dir.split('/')[0]:
        #     if is_lobster_graph(graphs[idx]):
        #         title_str += f' [L]'
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)


def save_fig(save_file=None, title='fig', dpi=300):
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        # fig_dir = os.path.join(*['samples', 'fig', save_dir])
        # if not os.path.exists(fig_dir):
        #     os.makedirs(fig_dir)
        plt.savefig(save_file,
                    bbox_inches='tight',
                    dpi=dpi,
                    transparent=False)
        plt.close()
    return


def save_graph_list(log_folder_name, exp_name, gen_graph_list):

    if not(os.path.isdir('./samples/pkl/{}'.format(log_folder_name))):
        os.makedirs(os.path.join('./samples/pkl/{}'.format(log_folder_name)))
    with open('./samples/pkl/{}/{}.pkl'.format(log_folder_name, exp_name), 'wb') as f:
            pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = './samples/pkl/{}/{}.pkl'.format(log_folder_name, exp_name)
    return save_dir


def plot_sampling_sweep(x_axis_data, y_axis_data, title, save_dir=None, x_label='x', y_label='y'):
    plt.figure()
    plt.plot(x_axis_data, y_axis_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    filename = title.split(' ')
    filename = '_'.join(filename)
    filename = filename.replace('/', '') + '.png'
    save_file = os.path.join(save_dir, filename)

    save_fig(save_file=save_file, title=title)


def parse_sweep_metrics(output_file):
    with open(output_file, 'r') as f:
        lines = f.readlines()

    nb_sampling_steps = []
    nspdks = []
    val_wo_corr = []
    val = []
    unique_at_1000 = []
    fcd_per_test = []
    novelity = []
    nspdk_mmd = []
    time = []

    'valid: '
    'unique@1000: '
    'FCD/Test: '
    'Novelty: '
    'NSPDK MMD: '
    'Generated 1000 molecules in '


    for line in lines:
        # no. of sampling steps
        if 'Total number of sampling steps:  ' in line:
            nb_sampling_steps.append(int(line.split('Total number of sampling steps:  ')[1]))
        elif 'nspdk' in line:
            value = line.split(':')[1]
            value = float('0.' + value.split('.')[1][0:6])
            nspdks.append(value)
        elif 'validity w/o correction: ' in line:
            val_wo_corr.append(float(line.split('validity w/o correction: ')[1]))
        elif 'valid: ' in line:
            val.append(float(line.split('valid: ')[1]))
        elif 'unique@500: ' in line:
            unique_at_1000.append(float(line.split('unique@500: ')[1]))
        elif 'FCD/Test: ' in line:
            fcd_per_test.append(float(line.split('FCD/Test: ')[1]))
        elif 'Novelty: ' in line:
            novelity.append(float(line.split('Novelty: ')[1]))
        elif 'NSPDK MMD: ' in line:
            nspdk_mmd.append(float(line.split('NSPDK MMD: ')[1]))
        elif 'Generated 500 molecules in ' in line:
            time.append(float(line.split('Generated 500 molecules in ')[1].split('s')[0]))

    print("Length of nb_sampling_steps: ", len(nb_sampling_steps))
    print("Length of nspdks: ", len(nspdks))
    print("Length of val_wo_corr: ", len(val_wo_corr))
    print("Length of val: ", len(val))
    print("Length of unique_at_1000: ", len(unique_at_1000))
    print("Length of fcd_per_test: ", len(fcd_per_test))
    print("Length of novelity: ", len(novelity))
    print("Length of nspdk_mmd: ", len(nspdk_mmd))
    print("Length of time: ", len(time))

    return nb_sampling_steps, nspdks, val_wo_corr, val, unique_at_1000, fcd_per_test, novelity, nspdk_mmd, time


if __name__ == "__main__":
    save_dir = '../plots/qm9_sweep_ckpt'
    nb_sampling_steps, nspdks, val_wo_corr, val, unique_at_1000, fcd_per_test, novelity, nspdk_mmd, time = parse_sweep_metrics('../qm9_sweep_ckpt.txt')

    print("len: ", len(nb_sampling_steps))

    plot_sampling_sweep(nb_sampling_steps, nspdks, 'NSPDK', save_dir=save_dir, x_label='No. of Sampling Steps', y_label='NSPDK')
    plot_sampling_sweep(nb_sampling_steps, val_wo_corr, 'Validity w/o Correction', save_dir=save_dir, x_label='No. of Sampling Steps', y_label='Validity w/o Correction')
    plot_sampling_sweep(nb_sampling_steps, val, 'Validity', save_dir=save_dir, x_label='No. of Sampling Steps', y_label='Validity')
    plot_sampling_sweep(nb_sampling_steps, unique_at_1000, 'Unique@1000', save_dir=save_dir, x_label='No. of Sampling Steps', y_label='Unique@1000')
    plot_sampling_sweep(nb_sampling_steps, fcd_per_test, 'FCD/Test', save_dir=save_dir, x_label='No. of Sampling Steps', y_label='FCD/Test')
    plot_sampling_sweep(nb_sampling_steps, novelity, 'Novelty', save_dir=save_dir, x_label='No. of Sampling Steps', y_label='Novelty')
    plot_sampling_sweep(nb_sampling_steps, nspdk_mmd, 'NSPDK MMD', save_dir=save_dir, x_label='No. of Sampling Steps', y_label='NSPDK MMD')
    plot_sampling_sweep(nb_sampling_steps, time, 'Time', save_dir=save_dir, x_label='No. of Sampling Steps', y_label='Time in Sec.')
