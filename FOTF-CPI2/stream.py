import pandas as pd
import torch
from torch.utils import data
import json
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs

vocab_path = './ESPF/vocab_protein.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/VOLT_dict_protein.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

vocab_path = './ESPF/vocab_drug.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/VOLT_dict_drug.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
# print(words2idx_d)

max_d = 50
max_p = 550


def protein2emb_encoder(x):
    # print(x[80:90])
    t1 = pbpe.process_line(x).split()  # split
    # print("protein", t1)
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p

    return i, np.asarray(input_mask)


def drug2emb_encoder(x):
    max_d = 40
    # print(x)
    # max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    # print("drug", t1)
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        # d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']

        # d_v = drug2single_vector(d)
        d_v, input_mask_d = drug2emb_encoder(d)
        p_v, input_mask_p = protein2emb_encoder(p)
        # print('demo')

        # print(d_v.shape)
        # print(input_mask_d.shape)
        # print(p_v.shape)
        # print(input_mask_p.shape)
        y = self.labels[index]
        return d_v, p_v, input_mask_d, input_mask_p, y

#
# # #
# def get_task(task_name):
#     if task_name.lower() == 'biosnap':
#         return './dataset/BIOSNAP/full_data'
#     elif task_name.lower() == 'bindingdb':
#         return './dataset/BindingDB'
#     elif task_name.lower() == 'davis':
#         return './dataset/DAVIS'
#
#
# params = {'batch_size': 32,
#           'shuffle': True,
#           'num_workers': 0,
#           'drop_last': True}
#
# dataFolder = get_task('bindingdb')
# #
# # df_train = pd.read_csv(dataFolder + '/train.csv')
# df_val = pd.read_csv(dataFolder + '/test_split.csv')
#
# validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
# validation_generator = data.DataLoader(validation_set, **params)
#
# # print(df_val['Target Sequence'][204])
#
# protein = 'LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPT'
#
# res_P = protein2emb_encoder(protein)
# res_D = drug2emb_encoder(df_val['SMILES'][204])
# # for i, (d, p, d_mask, p_mask, label) in enumerate(validation_generator):
# #     print(d.shape)
# #     print(d_mask.shape)
#
# # print(validation_set.df.keys())
# # print(validation_set.df["drug_encoding"])
# # print(validation_set.labels[51])
# # print(validation_set.list_IDs)
# #
# # for item in range(10010):
# #     k = validation_set[item]
# #
# # validation_generator = data.DataLoader(validation_set, **params)
#
# # for i, (d, p, d_mask, p_mask, label) in enumerate(validation_generator):
# #     if i == 55:
# #         print("i:", i)
# #         print(d)
# #         print(d_mask)
# #         print(p)
# # print(p)
# # print(p_mask)
