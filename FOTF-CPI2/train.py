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

torch.manual_seed(2)  # reproducible torch:2 np:3
np.random.seed(3)
from argparse import ArgumentParser
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='MolTrans Training.')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
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
    elif task_name.lower() == 'hybrid_data':
        return './dataset/Hybrid_data'
    elif task_name.lower() == 'dud_e':
        return './dataset/DUD_E'
    elif task_name.lower() == 'dude':
        return './dataset/DUDE'


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        # print(i)
        # print(d.dtype)
        # print(d.shape)
        # # if i == 110:
        # #     print(d)
        # print(p.dtype)
        # print(p.shape)

        score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
        # print("????")

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count
    # print(y_pred)

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    # print(thresholds)

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
                                                                                     outputs), y_pred, loss.item(), y_label


def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))


def main():
    config = BIN_config_DBPE()
    args = parser.parse_args()
    config['batch_size'] = args.batch_size

    loss_history = []

    model = BIN_Interaction_Flat(**config)

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('--- Data Preparation ---')
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    task = config["dataset_name"]
    # task = "BindingDB"
    dataFolder = get_task(task)
    print(dataFolder)

    df_train = pd.read_csv(dataFolder + '/train_split.csv')
    df_val = pd.read_csv(dataFolder + '/test_split.csv')
    df_test = pd.read_csv(dataFolder + '/test_split.csv')

    # print(df_train)
    # print(df_train.Label.values)

    training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
    print(validation_set.df.keys())

    validation_generator = data.DataLoader(validation_set, **params)
    print(validation_generator.dataset)

    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    # # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)

    print('--- Go for Training ---')

    torch.backends.cudnn.benchmark = True
    for epo in range(args.epochs):
        model.train()
        for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))

            loss = loss_fct(n, label)
            loss_history.append(loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i % 1000 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        # every epoch test
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss, label = test(testing_generator, model)
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc
                print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(
                    auprc) + ' , F1: ' + str(f1))
                with open(dataFolder + '/Label_best.pickle', 'wb') as f:
                    pickle.dump(label, f)
                with open(dataFolder + '/Pre_Label_best.pickle', 'wb') as f:
                    pickle.dump(logits, f)

    # model save

    save_model_dict(model_max, dataFolder, task)

    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss, label = test(testing_generator, model_max)
            print(
                'Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(
                    loss))
            with open(dataFolder + '/Label.pickle', 'wb') as f:
                pickle.dump(label, f)
            with open(dataFolder + '/Pre_Label.pickle', 'wb') as f:
                pickle.dump(logits, f)
    except:
        print('testing failed')
    return model_max, loss_history


s = time()
main()
e = time()
print(e - s)

# 2750/766

# 123/545
