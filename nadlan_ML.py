#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:41:04 2021

@author: shlomi
"""
features = ['ASSETROOMNUM', 'FLOORNO', 'DEALNATURE', 'NEWPROJECTTEXT',
            'BUILDINGYEAR', 'BUILDINGFLOORS', 'SEI_value', 'New', 'Ground',
            'TLV_proximity_value', 'PAI_value', 'P2015_value', 'year']

features1 = ['FLOORNO', 'DEALNATURE', 'NEWPROJECTTEXT',
            'BUILDINGYEAR',  'SEI_value', 'Ground', 'P2015_value', 'year']


def nadlan_simple_ML(df, year=2000, model_name='RF', feats=features1):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    X = df[[x for x in feats]]
    X = X.dropna()
    y = df['DEALAMOUNT']
    y = y.loc[X.index]
    if year is not None:
        X = X[X['year'] == year].drop('year', axis=1)
        y = y.loc[X.index]
    ml = ML_Classifier_Switcher()
    model = ml.pick_model(model_name)
    model.fit(X, y)
    if hasattr(model, 'feature_importances_'):
        di = dict(zip(X.columns, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        di = dict(zip(X.columns, model.coef_))
    dff = pd.DataFrame(di, index=[0]).T
    dff.columns = ['feature_importances']
    dff = dff.sort_values('feature_importances', ascending=False)
    dff.plot(kind='barh')
    plt.figure()
    sns.heatmap(X.corr(), annot=True)
    return X, y, model


def cross_validation(X, y, model_name='RF', n_splits=5, pgrid='light',
                     savepath=None, verbose=0, n_jobs=-1, year=None):
    from sklearn.model_selection import GridSearchCV
    ml = ML_Classifier_Switcher()
    model = ml.pick_model(model_name, pgrid=pgrid)
    param_grid = ml.param_grid
    gr = GridSearchCV(model, scoring='r2', param_grid=param_grid,
                      cv=n_splits, verbose=verbose, n_jobs=n_jobs)
    gr.fit(X, y)
    if savepath is not None:
        if year is not None:
            filename = 'GRSRCHCV_{}_{}_{}_{}.pkl'.format(model_name, n_splits,
                                                         pgrid, year)
        else:
            filename = 'GRSRCHCV_{}_{}_{}.pkl'.format(model_name, n_splits,
                                                      pgrid)
        save_gridsearchcv_object(gr, savepath, filename)
    return gr


def save_gridsearchcv_object(GridSearchCV, savepath, filename):
    import joblib
    print('{} was saved to {}'.format(filename, savepath))
    joblib.dump(GridSearchCV, savepath / filename)
    return
    # cvr = gr.cv_results_
    return GridSearchCV


def manual_cross_validation_for_RF_feature_importances(X, y, rf_model,
                                                       n_splits=5,
                                                       n_repeats=20,
                                                       scorers=['r2']):
    from sklearn.model_selection import KFold
    import xarray as xr
    import numpy as np
    from sklearn.model_selection import RepeatedKFold
    # from sklearn.metrics import make_scorer
    # scores_dict = {s: s for s in scorers}
    print(rf_model)
    # cv = TimeSeriesSplit(5)
    cv = KFold(5)  # , shuffle=False, random_state=42)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                       random_state=42)
    fis = []
    for train_index, test_index in cv.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rf_model.fit(X_train, y_train)
        fis.append(rf_model.feature_importances_)
    fi = xr.DataArray(fis, dims=['repeats', 'regressor'])
    fi['repeats'] = np.arange(1, len(fis)+1)
    fi['regressor'] = X.columns
    fi.name = 'feature_importances'
    return fi


def get_HP_params_from_optimized_model(path, pgrid='light', year=2000,
                                       model_name='RF', return_df=False):
    import joblib
    from Migration_main import path_glob
    files = path_glob(path, 'GRSRCHCV_{}_*_{}_{}.pkl'.format(model_name, pgrid, year))
    file = [x for x in files if model_name in x.as_posix()][0]
    gr = joblib.load(file)
    df = read_one_gridsearchcv_object(gr)
    if return_df:
        return df
    else:
        return df.iloc[0][:-2].to_dict()


def read_one_gridsearchcv_object(gr):
    """read one gridsearchcv multimetric object and
    get the best params, best mean/std scores"""
    import pandas as pd
    # first get all the scorers used:
    if isinstance(gr.scoring, list):
        scorers = [x for x in gr.scoring]
    elif isinstance(gr.scoring, str):
        scorers = ['score']
    # now loop over the scorers:
    best_params = []
    best_mean_scores = []
    best_std_scores = []
    for scorer in scorers:
        df_mean = pd.concat([pd.DataFrame(gr.cv_results_["params"]), pd.DataFrame(
            gr.cv_results_["mean_test_{}".format(scorer)], columns=[scorer])], axis=1)
        df_std = pd.concat([pd.DataFrame(gr.cv_results_["params"]), pd.DataFrame(
            gr.cv_results_["std_test_{}".format(scorer)], columns=[scorer])], axis=1)
        # best index = highest score:
        best_ind = df_mean[scorer].idxmax()
        best_mean_scores.append(df_mean.iloc[best_ind][scorer])
        best_std_scores.append(df_std.iloc[best_ind][scorer])
        best_params.append(df_mean.iloc[best_ind].to_frame().T.iloc[:, :-1])
    best_df = pd.concat(best_params)
    best_df['mean_score'] = best_mean_scores
    best_df['std_score'] = best_std_scores
    best_df.index = scorers
    return best_df


# def process_gridsearch_results(GridSearchCV, model_name,
#                                split_dim='inner_kfold', features=None,
#                                pwv_id=None, hs_id=None, test_size=None):
#     import xarray as xr
#     import pandas as pd
#     import numpy as np
#     # finish getting best results from all scorers togather
#     """takes GridSreachCV object with cv_results and xarray it into dataarray"""
#     params = GridSearchCV.param_grid
#     scoring = GridSearchCV.scoring
#     results = GridSearchCV.cv_results_
#     names = [x for x in params.keys()]
#     # unpack param_grid vals to list of lists:
#     pro = [[y for y in x] for x in params.values()]
#     ind = pd.MultiIndex.from_product((pro), names=names)
#     result_names = [
#         x for x in results.keys() if 'param' not in x]
#     ds = xr.Dataset()
#     for da_name in result_names:
#         da = xr.DataArray(results[da_name])
#         ds[da_name] = da
#     ds = ds.assign(dim_0=ind).unstack('dim_0')
#     for dim in ds.dims:
#         if ds[dim].dtype == 'O':
#             try:
#                 ds[dim] = ds[dim].astype(str)
#             except ValueError:
#                 ds = ds.assign_coords({dim: [str(x) for x in ds[dim].values]})
#         if ('True' in ds[dim]) and ('False' in ds[dim]):
#             ds[dim] = ds[dim] == 'True'
#     # get all splits data and concat them along number of splits:
#     all_splits = [x for x in ds.data_vars if 'split' in x]
#     train_splits = [x for x in all_splits if 'train' in x]
#     test_splits = [x for x in all_splits if 'test' in x]
#     return ds
#     # loop over scorers:
#     # trains = []
#     tests = []
#     for scorer in scoring:
#         train_splits_scorer = [x for x in train_splits if scorer in x]
#         # trains.append(xr.concat([ds[x]
#                                  # for x in train_splits_scorer], split_dim))
#         test_splits_scorer = [x for x in test_splits if scorer in x]
#         tests.append(xr.concat([ds[x] for x in test_splits_scorer], split_dim))
#         splits_scorer = np.arange(1, len(train_splits_scorer) + 1)
#     return tests
#     # train_splits = xr.concat(trains, 'scoring')
#     test_splits = xr.concat(tests, 'scoring')
#     return ds
# #    splits = [x for x in range(len(train_splits))]
# #    train_splits = xr.concat([ds[x] for x in train_splits], 'split')
# #    test_splits = xr.concat([ds[x] for x in test_splits], 'split')
#     # replace splits data vars with newly dataarrays:
#     ds = ds[[x for x in ds.data_vars if x not in all_splits]]
#     # ds['split_train_score'] = train_splits
#     ds['split_test_score'] = test_splits
#     ds[split_dim] = splits_scorer
#     if isinstance(scoring, list):
#         ds['scoring'] = scoring
#     elif isinstance(scoring, dict):
#         ds['scoring'] = [x for x in scoring.keys()]
#     ds.attrs['name'] = 'CV_results'
#     ds.attrs['param_names'] = names
#     ds.attrs['model_name'] = model_name
#     ds.attrs['{}_splits'.format(split_dim)] = ds[split_dim].size
#     if GridSearchCV.refit:
#         if hasattr(GridSearchCV.best_estimator_, 'feature_importances_'):
#             f_import = xr.DataArray(
#                 GridSearchCV.best_estimator_.feature_importances_,
#                 dims=['feature'])
#             f_import['feature'] = features
#             ds['feature_importances'] = f_import
#         ds['best_score'] = GridSearchCV.best_score_
# #        ds['best_model'] = GridSearchCV.best_estimator_
#         ds.attrs['refitted_scorer'] = GridSearchCV.refit
#         for name in names:
#             if isinstance(GridSearchCV.best_params_[name], tuple):
#                 GridSearchCV.best_params_[name] = ','.join(
#                     map(str, GridSearchCV.best_params_[name]))
#             ds['best_{}'.format(name)] = GridSearchCV.best_params_[name]
#         return ds, GridSearchCV.best_estimator_
#     else:
#         return ds, None


class ML_Classifier_Switcher(object):

    def pick_model(self, model_name, pgrid='normal'):
        """Dispatch method"""
        # from sklearn.model_selection import GridSearchCV
        self.param_grid = None
        method_name = str(model_name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid ML Model")
#        if gridsearch:
#            return(GridSearchCV(method(), self.param_grid, n_jobs=-1,
#                                return_train_score=True))
#        else:
        # Call the method as we return it
        # whether to select lighter param grid, e.g., for testing purposes.
        self.pgrid = pgrid
        return method()

    def SVM(self):
        from sklearn.svm import SVR
        import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'kernel': ['poly'],
                               'C': [0.1],
                               'gamma': [0.0001],
                               'degree': [1, 2],
                               'coef0': [1, 4]}
        # elif self.pgrid == 'normal':
        #     self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
        #                        'C': order_of_mag(-1, 2),
        #                        'gamma': order_of_mag(-5, 0),
        #                        'degree': [1, 2, 3, 4, 5],
        #                        'coef0': [0, 1, 2, 3, 4]}
        elif self.pgrid == 'dense':
            # self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
            #                    'C': np.logspace(-2, 2, 10), # order_of_mag(-2, 2),
            #                    'gamma': np.logspace(-5, 1, 14), # order_of_mag(-5, 0),
            #                    'degree': [1, 2, 3, 4, 5],
            #                    'coef0': [0, 1, 2, 3, 4]}
            self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear'],
                               'C': np.logspace(-2, 2, 10), # order_of_mag(-2, 2),
                                'gamma': np.logspace(-5, 1, 14)}#, # order_of_mag(-5, 0),
                               # 'degree': [1, 2, 3, 4, 5],
                               # 'coef0': [0, 1, 2, 3, 4]}
        return SVR()

    def MLP(self):
        import numpy as np
        from sklearn.neural_network import MLPRegressor
        if self.pgrid == 'light':
            self.param_grid = {
                'activation': [
                    'identity',
                    'relu'],
                'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50)]}
        # elif self.pgrid == 'normal':
        #     self.param_grid = {'alpha': order_of_mag(-5, 1),
        #                        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #                        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        #                        'learning_rate': ['constant', 'adaptive'],
        #                        'solver': ['adam', 'lbfgs', 'sgd']}
        elif self.pgrid == 'dense':
            self.param_grid = {'alpha': np.logspace(-5, 1, 7),
                               'activation': ['identity', 'logistic', 'tanh', 'relu'],
                               'hidden_layer_sizes': [(10, 10, 10), (10, 20, 10), (10,), (5,), (1,)],
                               'learning_rate': ['constant', 'adaptive'],
                               'solver': ['adam', 'lbfgs', 'sgd']}
            #(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
        return MLPRegressor(random_state=42, max_iter=500, learning_rate_init=0.1)

    def RF(self):
        from sklearn.ensemble import RandomForestRegressor
        # import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'max_features': ['auto', 'sqrt'],
                               'max_depth': [5, 10],
                               'min_samples_leaf': [1, 2],
                               'min_samples_split': [2, 5],
                               'n_estimators': [100, 300]}
        elif self.pgrid == 'normal':
            self.param_grid = {'max_depth': [5, 10, 25, 50, 100],
                               'max_features': ['auto', 'sqrt'],
                               'min_samples_leaf': [1, 2, 5, 10],
                               'min_samples_split': [2, 5, 15, 50],
                               'n_estimators': [100, 300, 700, 1200]
                               }
        elif self.pgrid == 'dense':
            self.param_grid = {'max_depth': [5, 10, 25, 50, 100, 150, 250],
                               'max_features': ['auto', 'sqrt'],
                               'min_samples_leaf': [1, 2, 5, 10, 15, 25],
                               'min_samples_split': [2, 5, 15, 30, 50, 70, 100],
                               'n_estimators': [100, 200, 300, 500, 700, 1000, 1300, 1500]
                               }
        return RandomForestRegressor(random_state=42, n_jobs=-1)

    def MLR(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(n_jobs=-1)
