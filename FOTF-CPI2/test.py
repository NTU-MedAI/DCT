# YZY
import copy
from time import time
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from torch import nn
from torch.autograd import Variable
from torch.utils import data
import pickle
from argparse import ArgumentParser
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream_fragement import BIN_Data_Encoder

torch.manual_seed(2)  # reproducible torch:2 np:3
np.random.seed(3)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='MolTrans Testing.')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--task', choices=['biosnap', 'bindingdb', 'davis'],
                    default='', type=str, metavar='TASK',
                    help='Task name. Could be biosnap, bindingdb and davis.')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')


def get_task(task_name):
    if task_name.lower() == 'biosnap':
        return './dataset/BIOSNAP'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'davis':
        return './dataset/DAVIS'


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    with torch.no_grad():
        for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
            score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
            # print("????")

            m = torch.nn.Sigmoid()
            logits = torch.squeeze(m(score))
            loss_fct = torch.nn.BCELoss()
            # print("fuck")

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            count += 1

            logits = logits.detach().cpu().numpy()
            # print("fuck")

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

    # print("??!!")
    loss = loss_accumulate / count
    # print("??!!")
    # print(y_pred)

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)
    # print(precision)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    # print(f1)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label,
                                                                   y_pred), f1_score(y_label,
                                                                                     outputs), y_pred_s, y_label


def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))


def main():
    config = BIN_config_DBPE()
    args = parser.parse_args()
    config['batch_size'] = args.batch_size

    print('--- Data Preparation ---')
    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': 0,
              'drop_last': True}

    task = "bindingdb"

    dataFolder = get_task(task)
    df_test = pd.read_csv(dataFolder + '/test_split.csv')
    # df_test = pd.read_csv(dataFolder + '/visualization_test_204.csv')
    # print(df_test["Label"][0:100])
    # df_test = df_test[0:31]
    # print(df_test[0:32])

    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test, 'test')
    testing_generator = data.DataLoader(testing_set, **params)

    print('--- Go for Testing ---')

    model = BIN_Interaction_Flat(**config).cuda()

    # model.load_state_dict(torch.load("dataset/BindingDB/bindingdb_fusion.pt"))
    model.load_state_dict(torch.load("dataset/BindingDB/bindingdb_fusion.pt"))
    ROC_AUC_score, Average_precision_score, F1_score, y_pre, label = test(testing_generator, model)

    print(y_pre.__len__())

    # with open('dataset/BindingDB/visualization_test_204_pre_score.pickle', 'wb') as f:
    #     pickle.dump(np_y_pre, f)

    # print(
    #     'Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) )


s = time()
main()
e = time()
print(e - s)
