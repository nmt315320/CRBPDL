from keras.layers import Dense, Convolution1D, Dropout, Input, Activation, Flatten,MaxPool1D,add, AveragePooling1D, Bidirectional,GRU,LSTM,Multiply,Activation, MaxPooling1D,TimeDistributed,AvgPool1D
from keras.layers.merge import Concatenate,concatenate
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse
import os
import time
import math
import tensorflow as tf
from Deal_Kmer import *
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
import sys
import gensim
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import logging
from keras_self_attention import SeqSelfAttention
from scipy import interp
import xlwt
from DProcess import convertRawToXY


gpu_id = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
tf_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=tf_config)

np.random.seed(4)

def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)
def defineExperimentPaths(basic_path, methodName, experimentID):
    experiment_name = methodName + '/' + experimentID
    MODEL_PATH = basic_path + experiment_name + '/model/'
    CHECKPOINT_PATH = basic_path + experiment_name + '/checkpoints/'
    RESULT_PATH = basic_path + experiment_name + '/results/'
    mk_dir(MODEL_PATH)
    mk_dir(CHECKPOINT_PATH)
    mk_dir(RESULT_PATH)
    return [MODEL_PATH, CHECKPOINT_PATH, RESULT_PATH]
def seq2ngram(seqs, k, s, wv):
    list = []
    print('need to n-gram %d lines' % len(seqs))

    for num, line in enumerate(seqs):
        if num < 3000000:
            line = line.strip()
            l = len(line) 
            list2 = []
            for i in range(0, l, s):
                if i + k >= l + 1:
                    break
                list2.append(line[i:i + k])
            list.append(convert_data_to_index(list2, wv))
    return list
    
def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data


def split_overlap_seq(seq):
    window_size = 101
    overlap_size = 20
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - 101)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - 101)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        bag_seqs.append(seq)
    else:
        if remain_ins > 10:
            new_size = end - overlap_size
            seq1 = seq[-new_size:]
            bag_seqs.append(seq1)
    return bag_seqs

def build_class_file(np, ng, class_file):
    with open(class_file, 'w') as outfile:
        outfile.write('label' + '\n')
        for i in range(np):
            outfile.write('1' + '\n')
        for i in range(ng):
            outfile.write('0' + '\n')

def circRNA2Vec(k, s, vector_dim, model, MAX_LEN, pos_sequences, neg_sequences):
    model1 = gensim.models.Doc2Vec.load(model)
    pos_list = seq2ngram(pos_sequences, k, s, model1.wv)
    neg_list = seq2ngram(neg_sequences, k, s, model1.wv)
    seqs = pos_list + neg_list

    X = pad_sequences(seqs, maxlen=MAX_LEN,padding='post')
    y = np.array([1] * len(pos_list) + [0] * len(neg_list))
    y = to_categorical(y)
    indexes = np.random.choice(len(y), len(y), replace=False)

    
    embedding_matrix = np.zeros((len(model1.wv.vocab), vector_dim))
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector    
            
    return X, y, embedding_matrix
    
    
def read_fasta_file(fasta_file):
    seq_dict = {}
    bag_sen = list()
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line[0]=='>': 
            name = line[1:] 
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()
    
    for seq in seq_dict.values():
        seq = seq.replace('T', 'U')
        bag_sen.append(seq)
        
    return np.asarray(bag_sen)


def Generate_Embedding(seq_posfile, seq_negfile, model):
    
    seqpos = read_fasta_file(seq_posfile)
    seqneg = read_fasta_file(seq_negfile)
        
    X, y, embedding_matrix = circRNA2Vec(10, 1, 30, model, 101, seqpos, seqneg)
    return X, y, embedding_matrix





def MultiScale(input):
    A = Convolution1D(filters=64, kernel_size=1, padding='same')(input)
    input_bn = BatchNormalization(axis=-1)(A)
    input_at = Activation('sigmoid')(input_bn)
    A = Dropout(0.4)(input_at)
    C = Convolution1D(filters=64, kernel_size=1, padding='same')(input)
    input_bn = BatchNormalization(axis=-1)(C)
    input_at = Activation('sigmoid')(input_bn)
    C = Dropout(0.4)(input_at)
    C = Convolution1D(filters=64, kernel_size=3, padding='same')(C)
    input_bn = BatchNormalization(axis=-1)(C)
    input_at = Activation('sigmoid')(input_bn)
    C = Dropout(0.4)(input_at)
    D = Convolution1D(filters=64, kernel_size=1, padding='same')(input)
    input_bn = BatchNormalization(axis=-1)(D)
    input_at = Activation('sigmoid')(input_bn)
    D = Dropout(0.4)(input_at)
    D = Convolution1D(filters=64, kernel_size=5, padding='same')(D)
    input_bn = BatchNormalization(axis=-1)(D)
    input_at = Activation('sigmoid')(input_bn)
    D = Dropout(0.4)(input_at)
    D = Convolution1D(filters=64, kernel_size=5, padding='same')(D)
    input_bn = BatchNormalization(axis=-1)(D)
    input_at = Activation('sigmoid')(input_bn)
    D = Dropout(0.4)(input_at)
    merge = Concatenate(axis=-1)([A, C, D])
    shortcut_y = Convolution1D(filters=192, kernel_size=1, padding='same')(input)
    shortcut_y = BatchNormalization()(shortcut_y)
    result = add([shortcut_y, merge])
    result = Activation('swish')(result)
    return result

def createModel(embedding_matrix):

    input_row_One_Hot = 101
    input_col_One_Hot = 5

    input_row_ANF_NCP = 101
    input_col_ANF_NCP = 9

    input_row_CKSNAP_NCP = 150
    input_col_CKSNAP_NCP = 17

    input_row_PSTNPss_NCP = 99
    input_col_PSTNPss_NCP = 25

    sequence_input = Input(shape=(101, 84), name='sequence_input')
    sequence = Convolution1D(filters=128, kernel_size=3, padding='same')(sequence_input)
    sequence = BatchNormalization(axis=-1)(sequence)
    sequence = Activation('swish')(sequence)
    profile_input = Input(shape=(101,), name='profile_input')
    embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                          weights=[embedding_matrix], trainable=False)(profile_input)
    profile = Convolution1D(filters=128, kernel_size=3, padding='same')(embedding)
    profile = BatchNormalization(axis=-1)(profile)
    profile = Activation('swish')(profile)
    main_input = Input(shape=(input_row_One_Hot, input_col_One_Hot),name='main_input')
    main = Convolution1D(filters=128, kernel_size=3, padding='same')(main_input)
    main = BatchNormalization(axis=-1)(main)
    main = Activation('swish')(main)
    input_A = Input(shape=(input_row_ANF_NCP, input_col_ANF_NCP), name='input_A')
    A = Convolution1D(filters=128, kernel_size=3, padding='same')(input_A)
    A = BatchNormalization(axis=-1)(A)
    A = Activation('swish')(A)
    input_C = Input(shape=(input_row_CKSNAP_NCP, input_col_CKSNAP_NCP))
    C = Convolution1D(filters=128, kernel_size=3, padding='same')(input_C)
    C = BatchNormalization(axis=-1)(C)
    C = Activation('swish')(C)
    input_P = Input(shape=(input_row_PSTNPss_NCP, input_col_PSTNPss_NCP))
    P = Convolution1D(filters=128, kernel_size=3, padding='same')(input_P)
    P = BatchNormalization(axis=-1)(P)
    P = Activation('swish')(P)
    mergeInput = concatenate([sequence, profile, main, A], axis=-1)
    overallResult = MultiScale(mergeInput)
    overallResult = AveragePooling1D(pool_size=5)(overallResult)
    overallResult = Dropout(0.3)(overallResult)
    overallResult = Bidirectional(GRU(120, return_sequences=True))(overallResult)
    overallResult = SeqSelfAttention(
        attention_activation='sigmoid',
        name='Attention',
    )(overallResult)
    overallResult = Flatten()(overallResult)
    overallResult = Dense(101, activation='swish')(overallResult)
    ss_output = Dense(2, activation='softmax', name='ss_output')(overallResult)
    return Model(inputs=[sequence_input, profile_input, main_input, input_A], outputs=[ss_output])



    
def main(parser):
    from AnalyseFASTA import analyseFixedPredict
    big_results = []
    al_result = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    all_fpr = []
    all_roc_auc = []
    global roc_auc_score
    wb = xlwt.Workbook(encoding='utf-8')
    results=[]
    protein = parser.RBPID
    model = './circRNA2Vec/circRNA2Vec_model'
    file_storage = './result/'
    seqpos_path='./Datasets/circRNA-RBP/' + protein + '/positive'
    seqneg_path ='./Datasets/circRNA-RBP/' + protein + '/negative'

    Kmer = dealwithdata1(protein)
    Embedding, dataY,  embedding_matrix = Generate_Embedding(seqpos_path, seqneg_path, model)

    pos_data, pos_ids, pos_poses = analyseFixedPredict(seqpos_path, window=20, label=1)
    neg_data, neg_ids, neg_poses = analyseFixedPredict(seqneg_path, window=20, label=0)

    train_All2 = pd.concat([pos_data, neg_data])
    train_data = train_All2
    train_All = train_data
    trainX_One_Hot, trainY_One_Hot = convertRawToXY(train_All.values, train_data.values, codingMode='ENAC')


    #####################################ANF_NCP_EIIP_Onehot#####################################
    trainX_ANF_NCP, trainY_ANF_NCP = convertRawToXY(train_All.values, train_data.values,
                                                    codingMode='ANF_NCP_EIIP_Onehot')


    #####################################CKSNAP_NCP_EIIP_Onehot#####################################
    trainX_CKSNAP_NCP, trainY_CKSNAP_NCP = convertRawToXY(train_All.values, train_data.values,
                                                          codingMode='CKSNAP_NCP_EIIP_Onehot')


    #####################################PSTNPss_NCP_EIIP_Onehot#####################################
    trainX_PSTNPss_NCP, trainY_PSTNPss_NCP = convertRawToXY(train_All.values, train_data.values,
                                                            codingMode='PSTNPss_NCP_EIIP_Onehot')


    indexes = np.random.choice(Kmer.shape[0],Kmer.shape[0], replace=False)
    training_idx, test_idx = indexes[:round(((Kmer.shape[0])/10)*8)], indexes[round(((Kmer.shape[0])/10)*8):]
    
    train_sequence, test_sequence = Kmer[training_idx, :, :], Kmer[test_idx, :, :]
    train_profile, test_profile = Embedding[training_idx, :], Embedding[test_idx, :]
    train_onehot, testonehot = trainX_One_Hot[training_idx, :, :], trainX_One_Hot[test_idx, :, :]
    train_PSTNPss_NCP,test_PSTNPss_NCP = trainX_PSTNPss_NCP[training_idx, :, :], trainX_PSTNPss_NCP[test_idx, :, :]
    train_CKSNAP_NCP,test_CKSNAP_NCP = trainX_CKSNAP_NCP[training_idx, :, :], trainX_CKSNAP_NCP[test_idx, :, :]
    train_ANF_NCP,test_ANF_NCP = trainX_ANF_NCP[training_idx, :, :], trainX_ANF_NCP[test_idx, :, :]
    train_label, test_label = dataY[training_idx, :], dataY[test_idx, :]

    batchSize = 50
    basic_path = file_storage + '/'
    methodName = protein

    logging.basicConfig(level=logging.DEBUG)
    sys.stdout = sys.stderr
    logging.debug("Loading data...")

    
    tprs=[]
    mean_fpr=np.linspace(0,1,100)

    test_y = test_label[:, 1]
    cv=5
    kf = KFold(cv, True)
    aucs = []
    Acc = []
    precision1 = []
    recall1 = []
    fscore1 = []
    i = 0
    als=time.time()
    ###########################################Adaboost+CNN:

    from multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN
    n_estimators = 10
    epochs = 1


    for train_index, eval_index in kf.split(train_label):

        train_X1 = train_sequence[train_index]
        train_X2 = train_profile[train_index]
        train_X3 = train_onehot[train_index]
        train_X4 = train_ANF_NCP[train_index]
        train_X5 = train_CKSNAP_NCP[train_index]
        train_X6 = train_PSTNPss_NCP[train_index]
        train_y = train_label[train_index]
        train_One_Hot=trainY_One_Hot[train_index]

        eval_X1 = train_sequence[eval_index]
        eval_X2 = train_profile[eval_index]
        eval_X3 = train_onehot[eval_index]
        eval_X4 = train_ANF_NCP[eval_index]
        eval_X5 = train_CKSNAP_NCP[eval_index]
        eval_X6 = train_PSTNPss_NCP[eval_index]
        eval_y = train_label[eval_index]
        eval_One_Hot = trainY_One_Hot[eval_index]

        print('training_network size is ', len(train_X1))
        print('validation_network size is ', len(eval_X1))
        print('training_network size is ', len(train_X2))
        print('validation_network size is ', len(eval_X2))

        [MODEL_PATH, CHECKPOINT_PATH, RESULT_PATH] = defineExperimentPaths(basic_path, methodName,
                                                                                     str(i))
        logging.debug("Loading network/training configuration...")
        best_model = 'Models/species.hdf5'


        model = createModel(embedding_matrix)
        logging.debug("Model summary ... ")
        model.count_params()
        model.summary()
        checkpoint_weight = CHECKPOINT_PATH + "weights.best.hdf5"
        if (os.path.exists(checkpoint_weight)):
            print ("load previous best weights")
            model.load_weights(checkpoint_weight, by_name=True)


        model.compile(optimizer='adam',
                      loss={'ss_output': 'categorical_crossentropy'},metrics = ['accuracy'])
        logging.debug("Running training...")
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []


        startTime = time.time()

        bdt_real_test_CNN = Ada_CNN(
            base_estimator=createModel(embedding_matrix),
            n_estimators=n_estimators,
            learning_rate=1,
            epochs=epochs)

        his=bdt_real_test_CNN.fit( model,train_X1, train_X2, train_X3, train_X4, train_y, batchSize)
        train_loss.append(his.history['loss'])
        val_loss.append(his.history['val_loss'])
        train_acc.append(his.history['acc'])
        val_acc.append(his.history['val_acc'])
        epoch = range(len(acc))
        plt.plot(epoch, acc, 'r', label='Training acc')  # 'bo'为画蓝色圆点，不连线
        plt.plot(epoch, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()  # 绘制图例，默认在右上角

        plt.figure()

        plt.plot(epoch, train_loss, 'r', label='Training loss')
        plt.plot(epoch, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
        test_real_errors_CNN = bdt_real_test_CNN.estimator_errors_[:]

        ss_y_hat_test = bdt_real_test_CNN.predict(test_sequence,  test_profile, testonehot,  test_ANF_NCP)

        print('\n Testing accuracy of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(
            accuracy_score(bdt_real_test_CNN.predict(test_sequence,  test_profile, testonehot,  test_ANF_NCP), test_label[:,1])))
        endTime = time.time()
        ftime = startTime - endTime
        ytrue = test_y
        ypred = ss_y_hat_test

        y_pred = np.argmax(ss_y_hat_test, axis=-1)
        mat = confusion_matrix(test_y,  ypred)
        tp = float(mat[0][0])
        fp = float(mat[1][0])
        fn = float(mat[0][1])
        tn = float(mat[1][1])

        aucc = roc_auc_score(ytrue, ypred)
        aucs.append(aucc)

        fpr,tpr,thresholds=roc_curve(ytrue,ypred)
        tprs.append(interp(mean_fpr,fpr,tpr))
        mean_tpr += interp(mean_fpr,fpr,tpr)
        mean_tpr[0]=0.0
        auc(fpr, tpr)
        roc_auc = auc(fpr,tpr)
        all_tpr.append(tpr.tolist())
        all_fpr.append(fpr.tolist())
        all_roc_auc.append(roc_auc)
        acc = accuracy_score(ytrue, ypred)
        Acc.append(acc)

        fc = f1_score(test_y, ypred)

        pos = int(tp + fn)
        neg = int(fp + tn)
        if (tp + fp) == 0:
            precision = 1
        else:
            precision = tp / (tp + fp)
        if (tp + fn) == 0:
            recall = 1
        else:
            recall = se = tp / (tp + fn)
        if (tn + fp) == 0:
            sp = 1
        else:
            sp = tn / (tn + fp)
        if se == 1 or sp == 1:
            gm = 1
        else:
            gm = math.sqrt(se * sp)
        f_measure = f_score = fc
        if (tp + fp) * (tn + fn) * (tp + fn) * (tn + fp) == 0:
            mcc = 1
        else:
            mcc = (tp * tn - fn * fp) / (math.sqrt((tp + fp) * (tn + fn) * (tp + fn) * (tn + fp)))

        precision = precision_score(ytrue, ypred)
        recall = recall_score(ytrue, ypred)
        fscore = f1_score(ytrue, ypred)
        precision1.append(precision)
        recall1.append(recall)
        fscore1.append(fscore)
        print('Accuracy: ', acc)
        print('AUC: ', auc)
        results.append([i, '%0.4f' % precision, '%0.4f' % recall, '%0.4f' % se, '%0.4f' % sp, '%0.4f' % gm,
                        f_measure, f_score, '%0.4f' % mcc, '%0.4f' % aucc, tp, fn, fp, tn, pos, neg, ftime])
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))
        i = i + 1
    ale= time.time()
    all_endtime = als-ale
    r = pd.DataFrame(results)
    r.to_csv(parser.storage+'result.csv')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr /= cv
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()
    mean_tpr = mean_tpr.tolist()
    mean_fpr = mean_fpr.tolist()

    ws = wb.add_sheet('r1')
    ws.write(0, 0, 'Mean FPR')
    ws.write(1, 0, 'Mean TPR')
    for i in range(0, len(mean_fpr)):
        ws.write(0, i + 1, mean_fpr[i])
        ws.write(1, i + 1, mean_tpr[i])
    ws.write(2, 0, 'Mean ROC Area: %0.4f' % mean_auc)
    count = 3
    for num in range(0, len(all_tpr)):
        fold = num + 1
        ws.write(count, 0, 'FPR fold ' + str(fold))
        ws.write(count + 1, 0, 'TPR fold ' + str(fold))
        # break
        for i in range(0, len(all_tpr[num])):
            ws.write(count, i + 1, all_fpr[num][i])
            ws.write(count + 1, i + 1, all_tpr[num][i])
        ws.write(count + 2, 0, 'ROC Area: %0.4f' % all_roc_auc[num])
        count += 3
    print('OK!')
    wb.save('ROC+' + '.xls')
    print("acid AUC: %.4f " % np.mean(aucs))
    print("acid ACC: %.4f " % np.mean(Acc))
    print("acid Precision: %.4f " % np.mean(precision1))
    print("acid Recall: %.4f " % np.mean(recall1))
    print("acid time: %.4f " % all_endtime)

    mean_tpr=np.mean(tprs,axis=0)
    mean_tpr[-1]=1.0

    al_result.append(['%0.4f' % np.mean(aucs), '%0.4f' % np.mean(Acc), '%0.4f' % np.mean(precision1), '%0.4f' % np.mean(recall1)])

    big_results.append(results)
    al_result.append(big_results)
def parse_arguments(parser):
    parser.add_argument('--RBPID', type=str,  default='WTAP')
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    main(args)
