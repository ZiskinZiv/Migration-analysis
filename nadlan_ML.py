#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:41:04 2021

@author: shlomi
"""
from MA_paths import work_david
ml_path = work_david / 'ML'
features = ['ASSETROOMNUM', 'FLOORNO', 'DEALNATURE', 'NEWPROJECTTEXT',
            'BUILDINGYEAR', 'BUILDINGFLOORS', 'SEI_value', 'New', 'Ground',
            'TLV_proximity_value', 'PAI_value', 'P2015_value', 'year','Building_Growth_Rate']

features1 = ['FLOORNO', 'DEALNATURE', 'NEWPROJECTTEXT',
             'BUILDINGYEAR',  'SEI_value', 'Ground', 'P2015_value', 'year', 'Building_Growth_Rate']

features2 = ['FLOORNO', 'DEALNATURE', 'NEWPROJECTTEXT',
             'SEI_value', 'Ground', 'year', 'Building_Growth_Rate']
apts = ['דירה', 'דירה בבית קומות']
apts_more = apts + ["קוטג' דו משפחתי", "קוטג' חד משפחתי", "דירת גן","בית בודד", "דירת גג","דירת גג (פנטהאוז)"]


def scale_df(df, scaler, cols=None):
    import pandas as pd
    print('using {} scaler.'.format(scaler.__repr__()))
    if cols is None:
        scaled_vals = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(scaled_vals)
        df_scaled.columns = df.columns
    else:
        print('scaling only {} cols.'.format(cols))
        df_sliced = df[cols]
        scaled_vals = scaler.fit_transform(df_sliced)
        df_scaled = pd.DataFrame(scaled_vals)
        df_scaled.columns = cols
        df_rest = df[[x for x in df.columns if x not in cols]]
        df_scaled = pd.concat([df_scaled, df_rest], axis=1)
        df_scaled = df_scaled[[x for x in df.columns]]
    return df_scaled, scaler


def produce_X_y(df, year=2015, y_name='Price', plot_Xcorr=True):
    import seaborn as sns
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.preprocessing import PowerTransformer
    # from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import pandas as pd
    # first, subset for apts and year:
    df = df[df['Type_of_asset'].isin(apts_more)]
    df = df[df['Sale_year']==year]
    # assume another_floor_1 is floor_number where floor_number=NaN:
    floor1 = df.loc[(~df['Another_floor_1'].isnull())]['Another_floor_1']
    df.loc[floor1.index, 'Floor_number'] = floor1.values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # now slice for the features:
    y = df[y_name]
    df = df[['Rooms_345', 'New', 'Floor_number', 'Periph_value', 'SEI_value_{}'.format(year), 'Inflow_rate_index']]
    df = df.rename({'Rooms_345': 'Rooms', 'SEI_value_{}'.format(year): 'SEI_value', 'New': 'New_apartment'}, axis=1)
    df = df.dropna()
    df['Rooms'] = df['Rooms'].astype(int)
    df['Floor_number'] = df['Floor_number'].astype(int)
    df['New_apartment'] = df['New_apartment'].astype(int)
    y = y.loc[df.index].reset_index(drop=True).dropna()
    X = df.reset_index(drop=True).loc[y.index]
    sns.heatmap(X.corr(), annot=True)
    # do onhotencoding on rooms:
    jobs_encoder = LabelBinarizer()
    jobs_encoder.fit(X['Rooms'])
    transformed = jobs_encoder.transform(X['Rooms'])
    rooms = pd.DataFrame(transformed)
    rooms.columns = ['3_Rooms', '4_Rooms', '5_Rooms']
    X = pd.concat([X, rooms], axis=1)
    X = X.drop(['Rooms'], axis=1)
    # scale Floor numbers:
    X['Floor_number'] = np.log(X['Floor_number']+1)
    # finally, scale y to log10 and X to minmax 0-1:
    Xscaler = MinMaxScaler()
    #yscaler = MinMaxScaler()
    # yscaler = PowerTransformer(method='yeo-johnson',standardize=True)
    y_scaled = y.apply(np.log10)
    # y_scaled = yscaler.fit_transform(y_scaled.values.reshape(-1,1))
    y = pd.DataFrame(y_scaled, columns=[y_name])
    X, scaler = scale_df(X, scaler=Xscaler)
    y = y[y_name]
    return X, y, Xscaler #, yscaler


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
    from sklearn.model_selection import KFold
    cv = KFold(n_splits, shuffle=True, random_state=1)
    ml = ML_Classifier_Switcher()
    model = ml.pick_model(model_name, pgrid=pgrid)
    param_grid = ml.param_grid
    gr = GridSearchCV(model, scoring='r2', param_grid=param_grid,
                      cv=cv, verbose=verbose, n_jobs=n_jobs)
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


def cross_validate_using_optimized_HP(X, y, estimator, model='RF', n_splits=5,
                                      n_repeats=20, scorers=['r2']):
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import KFold
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.model_selection import GroupShuffleSplit
    import pandas as pd
    # logo = LeaveOneGroupOut()
    # gss = GroupShuffleSplit(n_splits=20, test_size=0.1, random_state=1)
    # from sklearn.metrics import make_scorer
    # X = produce_X()
    # if add_MLR2:
    #     X = add_enso2_and_enso_qbo_to_X(X)
    #     print('adding ENSO^2 and ENSO*QBO')
    # y = produce_y()
    # X = X.sel(time=slice('1994', '2019'))
    # y = y.sel(time=slice('1994', '2019'))
    # groups = X['time'].dt.year
    # scores_dict = {s: s for s in scorers}
    # if 'r2_adj' in scorers:
    #     scores_dict['r2_adj'] = make_scorer(r2_adj_score)
    # if 'MLR' not in model:
    #     hp_params = get_HP_params_from_optimized_model(path, model)
    # ml = ML_Classifier_Switcher()
    # ml_model = ml.pick_model(model_name=model)
    # if 'MLR' not in model:
    #     ml_model.set_params(**hp_params)
    print(estimator)
    # cv = TimeSeriesSplit(5)
    cv = KFold(n_splits, shuffle=True, random_state=1)
    # cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                    random_state=1)
    # if strategy == 'LOGO-year':
    #     print('using LeaveOneGroupOut strategy.')
    cvr = cross_validate(estimator, X, y, scoring=scorers, cv=cv)
    df = pd.DataFrame(cvr)
    # elif strategy == 'GSS-year':
    #     print('using GroupShuffleSplit strategy.')
    #     cvr = cross_validate(ml_model, X, y, scoring=scores_dict, cv=gss,
    #                          groups=groups)
    # else:
    #     cvr = cross_validate(ml_model, X, y, scoring=scores_dict, cv=cv)
    return df


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
    # cv = KFold(5, shuffle=True, random_state=2)  # , shuffle=False, random_state=42)
    if n_repeats is not None:
        print('chose {} repeats'.format(n_repeats))
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                           random_state=42)
    else:
        cv = KFold(n_splits, shuffle=True, random_state=42)  # , shuffle=False, random_state=42)
    fis = []
    for train_index, test_index in cv.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rf_model.fit(X_train, y_train)
        if hasattr(rf_model, 'feature_importances_'):
            fis.append(rf_model.feature_importances_)
            name = 'feature_importances'
        elif hasattr(rf_model, 'coef_'):
            fis.append(rf_model.coef_)
            name = 'beta'
    fi = xr.DataArray(fis, dims=['repeats', 'regressor'])
    fi['repeats'] = np.arange(1, len(fis)+1)
    fi['regressor'] = X.columns
    fi.name = name
    # fi = sort_fi(fi)
    return fi


def compare_ml_models_years(df, ml_path=ml_path, pgrid='light',
                            model_name='RF', years=[2015, 2017]):
    cvrs = []
    fis = []
    for year in years:
        _, gr = load_HP_params_from_optimized_model(ml_path,
                                                    pgrid=pgrid,
                                                    model_name=model_name,
                                                    year=year)
        best = gr.best_estimator_
        X, y, Xscaler = produce_X_y(df, year=year, y_name='Price')
        cvr = cross_validate_using_optimized_HP(
            X, y, best)
        fi = manual_cross_validation_for_RF_feature_importances(X, y, best,
                                                                n_repeats=20,
                                                                n_splits=5)
        cvrs.append(cvr)
        fis.append(fi)

    cvr = concat_categories(cvrs[0], cvrs[1])
    beta = concat_categories(fis[0], fis[1])
    return cvr, beta


# def sort_fi(fi):
#     fi_mean = fi.mean('repeats')
#     dff = fi_mean.to_dataframe().sort_values('feature_importances')
#     da = dff.to_xarray()['feature_importances']
#     fi = fi.sortby(da)
#     return fi


def concat_categories(da1, da2, cat1_val=2015, cat2_val=2017, cat_name='Sale_year'):
    import xarray as xr
    from nadlan_EDA import convert_da_to_long_form_df
    import pandas as pd
    if isinstance(da1, xr.DataArray):
        df1 = convert_da_to_long_form_df(da1)

        df2 = convert_da_to_long_form_df(da2)
    else:
        df1 = da1
        df2 = da2
    df1[cat_name] = cat1_val
    df2[cat_name] = cat2_val
    df = pd.concat([df1, df2], axis=0)
    return df


def plot_feature_importances(fi, year=2017, mode='beta'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from nadlan_EDA import convert_da_to_long_form_df
    sns.set_theme(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(17, 5))
    if not isinstance(fi, pd.DataFrame):
        df = convert_da_to_long_form_df(fi, value_name='feature_importances')
        if mode != 'beta':
            df['feature_importances'] = df['feature_importances'] * 100
        sns.barplot(data=df, x='feature_importances', y='regressor', ci='sd', ax=ax)
    else:
        if mode != 'beta':
            fi['value'] = fi['value'] * 100
        sns.barplot(data=fi, x='value', y='regressor', hue='Sale_year', ci='sd')
    ax.grid(True)
    if mode == 'beta':
        ax.set_xlabel('Coefficiants')
    else:
        ax.set_xlabel('Feature Importances [%]')
    ax.set_ylabel('Regressors')
    if year is not None:
        fig.suptitle('Year {}'.format(year))
    fig.tight_layout()
    return fig


def plot_cvr(cvr, y='test_r2', x='Model'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    sns.set_theme(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(6, 6))
    g = sns.barplot(data=cvr, x=x, y=y, hue='Sale_year', ci='sd')
    # g.despine(left=True)
    plt.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlabel('Model')
    ax.set_ylabel(r'R$^2$')
    fig.tight_layout()
    return fig



def load_HP_params_from_optimized_model(path, pgrid='light', year=2000,
                                        model_name='RF', return_df=False, return_object=True):
    import joblib
    from Migration_main import path_glob
    files = path_glob(path, 'GRSRCHCV_{}_*_{}_{}.pkl'.format(model_name, pgrid, year))
    file = [x for x in files if model_name in x.as_posix()][0]
    gr = joblib.load(file)
    if return_object:
        df, gr = read_one_gridsearchcv_object(gr)
        return df, gr
    else:
        df = read_one_gridsearchcv_object(gr)
    if return_df:
        return df
    else:
        return df.iloc[0][:-2].to_dict()


def read_one_gridsearchcv_object(gr, return_object=True):
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
    if return_object:
        return best_df, gr
    else:
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


    def Ridge(self):
        from sklearn.linear_model import Ridge
        import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'alpha': np.logspace(-5, 5, 11),
                               'solver': ['auto', 'svd', 'sparse_cg', 'saga']}
        elif self.pgrid == 'dense':
            self.param_grid = {'alpha': np.logspace(-10, 10, 21),
                               'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

        return Ridge(normalize=False, random_state=1)
