from sklearn.model_selection import BaseCrossValidator
from CGRtools.preparer import CGRpreparer
from collections import defaultdict
from random import shuffle as r_shuffle
import numpy as np
from sklearn.utils.validation import indexable, check_array


class TransformationOut(BaseCrossValidator):
    def __init__(self, n_splits=5, shuffle=False):
        self.n_splits = n_splits
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):
        cgr = CGRpreparer()
        X, y, groups = indexable(X, y, groups)
        cgrs = [cgr.condense(r) for r in X]

        structure_condition = defaultdict(set)
        condition_structure = defaultdict(set)

        for structure, condition in zip(cgrs, groups):
            structure_condition[structure].add(condition)
            condition_structure[condition].add(structure)

        train_data = defaultdict(list)
        test_data = []

        for n, (structure, condition) in enumerate(zip(cgrs, groups)):
            train_data[structure].append(n)
            if len(condition_structure[condition]) > 1:
                test_data.append(n)

        structures_weight = [(x, len(y)) for x, y in train_data.items()]

        if self.n_splits > len(train_data):
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of transformations: %d."
                             % (self.n_splits, len(train_data)))


        train_len = {x: len(y) for x, y in train_data.items()}
        train_ind = list(train_data.keys())

        fold_mean_size = len(cgrs) // self.n_splits

        if self.shuffle:
            r_shuffle(structures_weight)
        fold_len = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        n = 0
        fold = {}
        for i in train_ind:
            if fold_len[n] + train_len[i] <= fold_mean_size:
                fold.setdefault(n, []).extend(train_data[i])
                fold_len[n] += train_len[i]
            elif n == self.n_splits - 1:
                break
            else:
                n += 1
                fold.setdefault(n, []).extend(train_data[i])
                fold_len[n] += train_len[i]

        for i in fold:
            train = []
            test = []
            for j in range(self.n_splits):
                if not j == i:
                    train.extend(fold[j])
                else:
                    for c in fold[j]:
                        if groups is not None:
                            if c in test_data:
                                test.append(c)
                        else:
                            test.append(c)
            yield np.array(train), np.array(test)

