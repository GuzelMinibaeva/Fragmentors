from functions import *
from CGRtools.files.RDFrw import RDFread
from CGRtools.CGRcore import CGRcore
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import csv
import os


def main():
    list_lam = [i / 10 for i in range(1, 11)] # [0.1, 0.2, 0.3 и так далее до 1]
    list_gam = [10 ** i for i in range(-2, 3)] # [0.01, 0.1, 1, 10, 100]
    list_c = [10 ** i for i in range(-4, 3)] # [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

    CGR = CGRcore(cgr_type='0')

    with open("test DB.rdf") as data_base: # <_io.TextIOWrapper name='test DB.rdf' mode='r' encoding='UTF-8'>
        graph_list = []
        conditions_list = []

        for reaction in RDFread(data_base):
            graph = CGR.getCGR(reaction)
            graph_list.append(graph)
            conditions_list.append(graph.meta)

    with open("sol.csv") as solv:
        reader = csv.reader(solv)
        next(solv)
        solv_list = []
        solv_const_list = []

        for row in reader:
            solv_list.append(row[0])
            solv_const_list.append([float(prop) for prop in row[1:]])
        solv_dict = dict(zip(solv_list, solv_const_list))

        for con_list in conditions_list:
            con_list['additive.1'] = solv_dict[con_list['additive.1'].lower()]

    matrices_struct_dict = structural_kernel_matrices(graph_list, list_lam, dumping=True)
    matrices_struct_cond_dict = dict_of_kernel_struct_con_matrices(list_lam, list_gam, graph_list, conditions_list, dumping=True)

    input = open('./dumps/structure_conditions_kernel_dict ({} reactions)'.format(len(graph_list)), 'rb')
    struct_cond_dict = pickle.load(input)
    input.close()

    start = timer()
    for lam in list_lam:
        for gam in list_gam:
            for c in list_c:
                y_exp = []
                y_pred = []
                start1 = timer()
                x = struct_cond_dict[lam][gam]
                kf = KFold(n_splits=5)
                y = pd.Series(float(i['logK']) for i in conditions_list)

                for train, test in kf.split(x):
                    y_train = y.as_matrix()[train]
                    x_train = x.as_matrix()[train, :][:, train]
                    x_test = x.as_matrix()[test, :][:, train]
                    y_test = y.as_matrix()[test]
                    model = svm.SVR(C=c, epsilon=0.1, kernel='precomputed')
                    model.fit(x_train, y_train)
                    y_predict = model.predict(x_test)
                    y_exp.extend(np.array(y_test).tolist())
                    y_pred.extend(y_predict)

                r_squared = r2_score(y_exp, y_pred)
                rmse = mt.sqrt(mean_squared_error(y_exp, y_pred))

                end1 = timer()
                time_per_model = end1 - start1
                print('\nlambda = {}, gamma = {}, C = {}, R squared = {}, RMSE = {}, time per model = {}'.format(
                    lam, gam, c, r_squared, rmse, time_per_model))
    end = timer()
    time_total = end - start
    print('\nTotal time = {}'.format(time_total))


if __name__ == '__main__':
    if not os.path.exists('./dumps'):
        os.mkdir('./dumps')
    main()
