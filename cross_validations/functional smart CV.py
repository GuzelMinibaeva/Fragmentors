import numpy as np
import pandas as pd
import pickle
import random
from math import sqrt
from itertools import product

from sklearn.svm import SVR
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import FeatureUnion

from CGRtools.files.RDFrw import RDFread
from CGRtools.preparer import CGRpreparer

from collections import Counter
from CIMtools.preprocessing.fragmentor import Fragmentor
from CIMtools.preprocessing.reference import MetaReference
from CIMtools.preprocessing.common import iter2array
from CIMtools.mbparser import MBparser

way = "solvent:solvents E2.csv"

methods = ['Kfold', 'naive_transformation', 'transformation_out', 'solvent_LOO', 'solvent_out'] # types of validations
databases = ['E2', 'SN2', 'da', 'tautomers'] # names of datasets

print('Процесс запущен')

def smart_validation(block, descX, descY, method, name):
    r2_param = {}
    rmse_param = {}
    for C in [10, 100, 1000, 10000, 100000]:
        for gamma in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
            y_true, y_pred = [], []
            test_len = 0
            rep = 0
            r2rep, rmserep = [], []
            for train, test in block:
                test_len += len(test)
                if not method in ['naive_solv', 'solv_out']:

                    svr = SVR(gamma=gamma, C=C)
                    svr.fit(descX.loc[train], descY.loc[train])
                    test_pred = svr.predict(descX.loc[test])
                    y_true.append(descY.loc[test])
                    y_pred.append(test_pred)

                    rep += 1
                    if rep == 5:
                        y_true = pd.concat(y_true)
                        all_pred = np.array([])
                        for x in y_pred:
                            all_pred = np.concatenate((all_pred, x), axis=0)
                        y_pred = pd.Series(all_pred, index=y_true.index)

                        r2 = r2_score(y_pred, y_true)
                        rmse = sqrt(mean_squared_error(y_true, y_pred))
                        rmserep.append(rmse)
                        r2rep.append(r2)
                        y_true, y_pred = [], []
                        test_len = 0
                        rep = 0
                else:
                    svr = SVR(gamma=gamma, C=C)
                    svr.fit(descX.loc[train], descY.loc[train])
                    test_pred = svr.predict(descX.loc[test])
                    y_true.append(descY.loc[test])
                    y_pred.append(test_pred)

            if method in ['naive_solv', 'solv_out']:
                y_true = pd.concat(y_true)
                all_pred = np.array([])
                for x in y_pred:
                    all_pred = np.concatenate((all_pred, x), axis=0)
                y_pred = pd.Series(all_pred, index=y_true.index)
                r2 = r2_score(y_pred, y_true)
                rmse = sqrt(mean_squared_error(y_true, y_pred))
                rmserep.append(rmse)
                r2rep.append(r2)
            r2_param[(C, gamma)] = np.mean(r2rep)
            rmse_param[(C, gamma)] = np.mean(rmserep)

    for estimation, e_name in zip([r2_param, rmse_param], ['R2', 'RMSE']):
        l1 = []
        l2 = []
        res = []
        for x, y in estimation.items():
            l1.append(x[0])
            l2.append(x[1])
            res.append(y)
            array = [np.array(l1), np.array(l2)]
            tuples = list(zip(*array))
            index = pd.MultiIndex.from_tuples(tuples, names=['C', 'gamma'])
            table = pd.Series(res, index=index)
            table.to_excel(name + '_' + method + e_name +'.xlsx')

    with open(name + '_' + method +'_r2_.pickle', 'wb') as f:
        pickle.dump(r2_param, f)
    with open(name + '_' + method + '_rmse.pickle', 'wb') as f:
        pickle.dump(rmse_param, f)
    print(name+method +' is over')

for name in databases: # берем первый дасасет

    if name == 'tautomers':
        solvent_name = 'solvent' # растворитель
        constant_name = 'tabulated_constant' # s_options
    elif name == 'da' or name == 'SN2' or name == 'E2':
        solvent_name = 'additive.1'
        constant_name = 'logK'

    cgr = CGRpreparer() # инициализируем CGRpreparer()
    with open(name + '.rdf', 'r') as f:
        z = RDFread(file=f)
        x = z.read()
    cc = Counter(cgr.condense(r) for r in x)
    rc = list(cc.keys())

    reaction_solvent = {}
    solvent_reaction = {}
    for condition in x:
        rr = rc.index(cgr.condense(condition))
        tmp = reaction_solvent.setdefault(rr, {})
        if condition['meta'][solvent_name] in tmp:
            tmp[condition['meta'][solvent_name]] += 1
        else:
            tmp[condition['meta'][solvent_name]] = 1
        solvent_reaction.setdefault(condition['meta'][solvent_name], set()).add(rr)

    train_list_struct, trainsetC = {}, {}
    testsetS, testsetC, testsetV = [], [], []
    structlist = []
    react_ind = []
    varnek_list = []
    solvents = []
    for n, condition in enumerate(x):
        rs = cgr.condense(condition)
        r = rc.index(rs)
        s = condition['meta'][solvent_name]
        t = condition['meta']['temperature']
        tc = condition['meta'][constant_name]
        rs.meta.update(solvent=s, temperature=t, tabulated_constant=tc, reaction=r)
        structlist.append(rs)
        react_ind.append(n)

        if len(solvent_reaction[s]) > 1 or len(reaction_solvent[r]) > 1:
            train_list_struct.setdefault(r, []).append(n)
        if len(solvent_reaction[s]) > 1:
            testsetS.append(n)

        if len(solvent_reaction[s]) > 1 or len(reaction_solvent[r]) > 1:
            trainsetC.setdefault(s, []).append(n)
            solvents.append(s)
        if len(solvent_reaction[s]) > 1 and len(reaction_solvent[r]) > 1:
            testsetC.append(n)

        if reaction_solvent[r][s] == 1 and len(reaction_solvent[r]) == 1:
            varnek_list.append(n)

    extension = ["temperature", way]
    dd = MetaReference(data=MBparser.parse_ext(extension))
    df = Fragmentor(version='2017.x', fragment_type=3, min_length=2, max_length=6,
                    cgr_dynbonds=1, doallways=False, useformalcharge=False)

    descriptors = FeatureUnion([('Fragmetor', df), ('MetaReference', dd)])
    descX = iter2array(structlist)
    descXm = descriptors.fit_transform(descX)
    scaler = MinMaxScaler()
    descX = scaler.fit_transform(descXm)

    descY = np.array([float(m.meta[constant_name]) for m in structlist])

    descX = pd.DataFrame(descX)
    descY = pd.Series(descY)

    size = len(structlist) // 5 + 1

    train_len = {x: len(y) for x, y in train_list_struct.items()}
    train_ind = list(train_list_struct.keys())

    difference = 1.05
    fold_max_train = int(difference * size)

    for method in methods:
        if method == 'Kfold':
            rep = 0
            naive_block = []
            while True:
                random.shuffle(react_ind)
                folds = {}
                block = []
                for i in range(5):
                    if not len(structlist) // 5 == 0 and i == 5:
                        folds[i] = react_ind[i * size:(i + 1) * size - len(structlist) % 5]
                    else:
                        folds[i] = react_ind[i * size:(i + 1) * size]
                for i in range(5):
                    block.append([[]])
                    for j in range(5):
                        if i == j:
                            continue
                        else:
                            block[i][0].extend(folds[j])
                    print(len(block[i][0]))
                    print(len(folds[i]))
                    block[i].append(folds[i])
                naive_block.extend(block)
                rep += 1
                if rep > 4:
                    break
            print('Naive CV complete!')

            smart_validation(block=naive_block, descX=descX, descY=descY, method=method, name=name)

            r2_param = {}
            rmse_param = {}
            for C in [10, 100, 1000, 10000, 100000]:
                for gamma in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
                    y_true, y_pred = [], []
                    test_len = 0
                    rep = 0
                    r2rep, rmserep = [], []
                    for train, test in naive_block:

                        test_len += len(test)

                        svr = SVR(gamma=gamma, C=C)
                        svr.fit(descX.loc[train], descY.loc[train])
                        test_pred = svr.predict(descX.loc[test])
                        y_true.append(descY.loc[test])
                        y_pred.append(test_pred)

                        rep += 1
                        if rep == 5:
                            y_true = pd.concat(y_true)
                            z = np.array([])
                            for x in y_pred:
                                z = np.concatenate((z, x), axis=0)
                            y_pred = pd.Series(z, index=y_true.index)
                            tmp_true, tmp_pred = [], []
                            for react in varnek_list:
                                tmp_true.append(y_true.loc[react])
                                tmp_pred.append(y_pred.loc[react])
                            y_true = tmp_true
                            y_pred = tmp_pred

                            r2 = r2_score(y_pred, y_true)
                            rmse = sqrt(mean_squared_error(y_true, y_pred))
                            rmserep.append(rmse)
                            r2rep.append(r2)
                            y_true, y_pred = [], []
                            test_len = 0
                            rep = 0
                        r2_param[(C, gamma)] = np.mean(r2rep)
                        rmse_param[(C, gamma)] = np.mean(rmserep)
                with open(name + '_varnek_r2.pickle', 'wb') as f:
                    pickle.dump(r2_param, f)
                with open(name + '_varnek_rmse.pickle', 'wb') as f:
                    pickle.dump(rmse_param, f)
            print(name+'Валидация по Варнеку закончена')

        if method == 'transformation_out':
            rep = 0
            flag = True
            structure_block = []
            while True:
                random.shuffle(train_ind)
                block = []
                block_mass = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                n = 0
                fold = {}
                for i in train_ind:
                    if block_mass[n] + train_len[i] <= fold_max_train:
                        fold.setdefault(n, []).extend(train_list_struct[i])
                        block_mass[n] += train_len[i]
                    elif n == 4:
                        flag = False
                    else:
                        n += 1
                        fold.setdefault(n, []).extend(train_list_struct[i])
                        block_mass[n] += train_len[i]

                for i in fold:
                    block.append([[], []])
                    for j in range(5):
                        if not j == i:
                            block[i][0].extend(fold[j])
                        else:
                            for c in fold[j]:
                                if c in testsetS:
                                    block[i][1].append(c)

                if flag and sqrt(sum((x - size) ** 2 for x in block_mass.values()) / 5) < fold_max_train:
                    structure_block.extend(block)
                    print(block_mass, sum(x for x in block_mass.values()))
                    rep += 1
                    if rep > 4:
                        break
            print('Structure-out CV complete!')

            smart_validation(block=structure_block, descX=descX, descY=descY, method=method, name=name)

        if method == 'naive_transformation':
            stupid_transformation = []
            rep = 0
            flag = True
            while True:
                random.shuffle(train_ind)
                block = []
                block_mass = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                n = 0
                fold = {}
                for i in train_ind:
                    if block_mass[n] + train_len[i] <= fold_max_train:
                        fold.setdefault(n, []).extend(train_list_struct[i])
                        block_mass[n] += train_len[i]
                    elif n == 4:
                        flag = False
                    else:
                        n += 1
                        fold.setdefault(n, []).extend(train_list_struct[i])
                        block_mass[n] += train_len[i]

                for i in fold:
                    block.append([[], []])
                    for j in range(5):
                        if not j == i:
                            block[i][0].extend(fold[j])
                        else:
                            for c in fold[j]:
                                block[i][1].append(c)

                if flag and sqrt(sum((x - size) ** 2 for x in block_mass.values()) / 5) < fold_max_train:
                    stupid_transformation.extend(block)
                    print(block_mass, sum(x for x in block_mass.values()))
                    rep += 1
                    if rep > 4:
                        break

            smart_validation(block=stupid_transformation, descX=descX, descY=descY, method=method, name=name)


        if method == 'solvent_out':
            solvent_block = []
            solvent_names = []
            excluded_solvents = []

            for s, r in trainsetC.items():
                train = []
                test = []
                solvent_names.append(s)
                for rr in r:
                    if rr in testsetC:
                        test.append(rr)
                if len(test) == 0:
                    excluded_solvents.append(s)
                    continue
                for ss, rs in trainsetC.items():
                    if ss == s:
                        continue
                    else:
                        train.extend(rs)
                solvent_block.append([train, test])
            print('Solvent-out CV complete!')

            smart_validation(block=solvent_block, descX=descX, descY=descY, method=method, name=name)


        if method == 'solvent_LOO':
            loo = []
            for s, r in trainsetC.items():
                train = []
                test = []
                for rr in r:
                    test.append(rr)
                if len(test) == 0:
                    continue
                for ss, rs in trainsetC.items():
                    if ss == s:
                        continue
                    else:
                        train.extend(rs)
                loo.append([train, test])
            print('Leave-one-out CV complete!')

            smart_validation(block=loo, descX=descX, descY=descY, method=method, name=name)

