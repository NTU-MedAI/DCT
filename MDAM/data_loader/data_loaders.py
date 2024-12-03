import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from module.molgraph_data import drug2emb_encoder
from module.molgraph_data import smile_to_graph
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class QM93DDataLoader():

    def __init__(self,
                 root='/home/ntu/PycharmProjects/CY/Data/Original/',
                 train_batch_size=32,
                 valtest_batch_size=32,
                 train_size=1.1e5,valid_size=1e4):
        train_size = 110000
        valid_size = 10000
        self.root = root
        self.dataset = QM93D(root = self.root)
        split_idx = self.dataset.get_idx_split(len(self.dataset.data.y),
                                train_size=train_size, valid_size=valid_size, seed=42)
        self.train_dataset, self.valid_dataset, self.test_dataset = \
            self.dataset[split_idx['train']], self.dataset[split_idx['valid']], self.dataset[split_idx['test']]
        self.train_loader = DataLoader(self.train_dataset, train_batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, valtest_batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, valtest_batch_size, shuffle=False)




class QM93D(InMemoryDataset):
    r"""
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`QM9` dataset
        which is from `"Quantum chemistry structures and properties of 134 kilo molecules" <https://www.nature.com/articles/sdata201422>`_ paper.
        It connsists of about 130,000 equilibrium molecules with 12 regression targets:
        :obj:`mu`, :obj:`alpha`, :obj:`homo`, :obj:`lumo`, :obj:`gap`, :obj:`r2`, :obj:`zpve`, :obj:`U0`, :obj:`U`, :obj:`H`, :obj:`G`, :obj:`Cv`.
        Each molecule includes complete spatial information for the single low energy conformation of the atoms in the molecule.

        .. note::
            We used the processed data in `DimeNet <https://github.com/klicperajo/dimenet/tree/master/data>`_, wihch includes spatial information and type for each atom.
            You can also use `QM9 in Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9>`_.


        Args:
            root (string): the dataset folder will be located at root/qm9.
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        Example:
        --------

        >>> dataset = QM93D()
        >>> target = 'mu'
        >>> dataset.data.y = dataset.data[target]
        >>> split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
        >>> train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> data = next(iter(train_loader))
        >>> data
        Batch(Cv=[32], G=[32], H=[32], U=[32], U0=[32], alpha=[32], batch=[579], gap=[32], homo=[32], lumo=[32], mu=[32], pos=[579, 3], ptr=[33], r2=[32], y=[32], z=[579], zpve=[32])

        Where the attributes of the output data indicates:

        * :obj:`z`: The atom type.
        * :obj:`pos`: The 3D position for atoms.
        * :obj:`y`: The target property for the graph (molecule).
        * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs
    """
    def __init__(self, root='/home/ntu/PycharmProjects/CY/Data/Original/',
                 keeprows_dir = '/home/ntu/PycharmProjects/CY/Data/Original/Important/keeprows.npy',
                 csv_dir = '/home/ntu/PycharmProjects/CY/Data/Original/Important/qm9.csv',
                 transform=None, pre_transform=None, pre_filter=None):

        self.url = 'https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz'
        self.folder = osp.join(root, 'qm9')


        keeprows = list(np.load(keeprows_dir))
        self.smiles_data = list(pd.read_csv(csv_dir).iloc[keeprows]['smiles'])

        super(QM93D, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def processed_file_names(self):
        return 'qm9_pyg.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):

        data = np.load(osp.join(self.raw_dir, self.raw_file_names))


        R = data['R']
        Z = data['Z']
        N = data['N']
        split = np.cumsum(N)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z, split)
        target = {}
        for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']:
            target[name] = np.expand_dims(data[name], axis=-1)
        # y = np.expand_dims([data[name] for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']], axis=-1)

        data_list = []
        for i in tqdm(range(len(N))):

            adj_1, nd_1, ed_1 = smile_to_graph(self.smiles_data[i])
            d1, mask_1 = drug2emb_encoder(self.smiles_data[i])

            adj_1 = torch.tensor(np.array(adj_1, dtype=np.int64))
            adj = torch.zeros((9, 9))
            adj[:adj_1.shape[0], :adj_1.shape[1]] = adj_1

            nd_1 = torch.tensor(np.array(nd_1, dtype=np.int64))
            nd = torch.zeros((9, 75))
            nd[:nd_1.shape[0], :nd_1.shape[1]] = nd_1

            ed_1 = torch.tensor(np.array(ed_1, dtype=np.int64))
            ed = torch.zeros((9, 9, 4))
            ed[:ed_1.shape[0], :ed_1.shape[1], :] = ed_1

            d = torch.tensor(np.array(d1, dtype=np.int64))
            mask = torch.tensor(np.array(mask_1, dtype=np.int64))


            R_i = torch.tensor(R_qm9[i], dtype=torch.float32)
            z_i = torch.tensor(Z_qm9[i], dtype=torch.int64)
            y_i = [torch.tensor(target[name][i], dtype=torch.float32) for name in
                   ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']]
            data = Data(pos=R_i, z=z_i, y=y_i[0], mu=y_i[0], alpha=y_i[1], homo=y_i[2], lumo=y_i[3], gap=y_i[4],
                        r2=y_i[5], zpve=y_i[6], U0=y_i[7], U=y_i[8], H=y_i[9], G=y_i[10], Cv=y_i[11],
                        adj=adj, nd=nd, ed=ed, d=d,mask=mask)
            data_list.append(data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)

        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict