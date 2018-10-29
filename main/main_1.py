import numpy as np
from CIMtools.preprocessing import Fragmentor, CGR
from CIMtools.preprocessing.reference import prepare_metareference
# from CIMtools.descriptors import DescriptorsDict, Fragmentor, DescriptorsChain
# from CIMtools.mbparser import MBparser
from CGRtools.files.RDFrw import RDFread
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

# Name_of_file = ['Tautomres_new_739.rdf', 'DA_25.04.2017_All.rdf', 'SN2_article_2018_final++.rdf',
#                 'E2_26.04.18.rdf', 'SN1.rdf']
Name_of_file = ['/users/asima/home/asima/FR/datasets/Tautomres_new_739.rdf']

kf = KFold(n_splits=5, random_state=1, shuffle=True)

R2 = [[] for _ in range(5)]
RMSE = [[] for _ in range(5)]
index = [30, 31, 32, 91, 92, 93]

for num, i in enumerate(Name_of_file):
    print(i, num)

    if i == '/users/asima/home/asima/FR/datasets/Tautomres_new_739.rdf' or i == 'E2_26.04.18.rdf':
        s_option = 'tabulated_constant'
    elif i == 'DA_25.04.2017_All.rdf' or i== 'SN2_article_2018_final++.rdf':
        s_option = 'logK'
    else:
        s_option = 'logk'

    with open(i, encoding='cp1251') as f:
        reactions = RDFread(f).read()
    Y = np.array([float(reaction.meta[s_option]) for reaction in reactions])
    for i, j in product([3, 9], ['0', '1', '2']):
        fr = Fragmentor(version='2017.x', fragment_type=i, min_length=2, max_length=4, useformalcharge=1)
        meta = ['additive.1:/users/asima/home/asima/FR/datasets/solvents_uniq_table.csv', 'temperature:=1/x']
        md = prepare_metareference(meta)
        cgr = CGR(cgr_type=j)
        pipe = Pipeline([('cgr', cgr), ('frg', fr), ('scale', StandardScaler())])
        pipe_uniion = FeatureUnion([('cgr_frg', pipe), ('solv', md)])
        X = pipe_uniion.fit_transform(reactions)
        #
        # Y_pred = [[] for _ in range(5)]
        # Y_test = [[] for _ in range(5)]
        # for train, test in kf.split(x):
        #     x_train, x_test = x[train], x[test]
        #     y_train, y_test = Y[train], Y[test]
        #     est = GridSearchCV(RandomForestRegressor(random_state=1, n_estimators=500),
        #                        {'max_features': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
        #                                          0.35, 'auto', 'log2', None]},
        #                        cv=kf, verbose=1, scoring='neg_mean_squared_error', n_jobs=4)
        #     est.fit(x_train, y_train)
        #     y_pred = est.predict(x_test)
        #     Y_test[num].extend(y_test)
        #     Y_pred[num].extend(y_pred)
        # R2_score = r2_score(np.array(Y_test[num]), np.array(Y_pred[num]))
        # R2[num].append(R2_score)
        # RMSE_score = np.sqrt(mean_squared_error(Y_test[num], Y_pred[num]))
        # RMSE[num].append(RMSE_score)
        #
        # r2_param = {}
        # rmse_param = {}
        # for C in [10, 100, 1000, 10000, 100000]:
        #     for gamma in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        #         y_true, y_pred = [], []
        #         test_len = 0
        #         rep = 0
        #         r2rep, rmserep = [], []
        #         for train, test in naive_block:
        #
        #             test_len += len(test)
        #
        #             svr = SVR(gamma=gamma, C=C)
        #             svr.fit(descX.loc[train], descY.loc[train])
        #             test_pred = svr.predict(descX.loc[test])
        #             y_true.append(descY.loc[test])
        #             y_pred.append(test_pred)
        #
        #             rep += 1
        #             if rep == 5:
        #                 y_true = pd.concat(y_true)
        #                 z = np.array([])
        #                 for x in y_pred:
        #                     z = np.concatenate((z, x), axis=0)
        #                 y_pred = pd.Series(z, index=y_true.index)
        #                 tmp_true, tmp_pred = [], []
        #                 for react in varnek_list:
        #                     tmp_true.append(y_true.loc[react])
        #                     tmp_pred.append(y_pred.loc[react])
        #                 y_true = tmp_true
        #                 y_pred = tmp_pred
        #
        #                 r2 = r2_score(y_pred, y_true)
        #                 rmse = sqrt(mean_squared_error(y_true, y_pred))
        #                 rmserep.append(rmse)
        #                 r2rep.append(r2)
        #                 y_true, y_pred = [], []
        #                 test_len = 0
        #                 rep = 0
        #             r2_param[(C, gamma)] = np.mean(r2rep)
        #             rmse_param[(C, gamma)] = np.mean(rmserep)
        #     with open(name + '_varnek_r2.pickle', 'wb') as f:
        #         pickle.dump(r2_param, f)
        #     with open(name + '_varnek_rmse.pickle', 'wb') as f:
        #         pickle.dump(rmse_param, f)
        # print(name + 'Валидация по Варнеку закончена')