import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TKAgg')


def read_data(path: str = ''):
    # url = "https://github.com/zhentaoshi/Econ5821/raw/main/data_example/dataset_inf.Rdata"
    cpi = pd.read_csv(path + 'cpi.csv')
    ppi = pd.read_csv(path + 'ppi.csv')
    x = pd.read_csv(path + 'X.csv', encoding='gb2312')
    fake_testing_x = pd.read_csv(path + 'fake_testing_X.csv', encoding='gb2312')
    return cpi, ppi, x, fake_testing_x


def inflation_rate(data, name='CPI'):
    inflation_rate = pd.DataFrame(columns=['month', 'inflation_rate'])
    i = -1
    for index, row in data.iterrows():
        i += 1
        if i < 12:
            continue
        infla_rate = np.log(row[name]) - np.log(data.loc[index-12, name])
        inflation_rate.loc[len(inflation_rate.index)] = [row['month'], infla_rate]
    return inflation_rate


def evaluate_training_data(y_train, y_train_pred, y_test, y_test_pred, **kwargs):
    plt.scatter(y_train_pred,  y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.title('Residual errors')
    plt.hlines(y=0, xmin=-1, xmax=1, color='black', lw=2)
    plt.xlim([-0.25, 0.25])
    plt.tight_layout()
    plt.show()


def train_and_evaluate(model, x_train, x_test, y_train, y_test, plot=False, **kwargs):
    model.fit(x_train, y_train)
    y_train_pred, y_test_pred = model.predict(x_train), model.predict(x_test)
    if plot:
        evaluate_training_data(y_train, y_train_pred, y_test, y_test_pred, **kwargs)
    print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                           mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


def RegularizedRegression(x_train, x_test, y_train, y_test, model='lasso', begin=0.01, end=1.0, seq=0.01, plot=True,
                          l1_ratio=0.5, **kwargs):
    krange = np.arange(begin, end, seq)
    train_scores, test_scores = [], []
    if model == 'lasso':
        api = Lasso
    elif model == 'ridge':
        api = Ridge
    else:
        api = ElasticNet
    for k in krange:
        model_ = api(alpha=k, l1_ratio=l1_ratio) if model == 'elasticnet' else api(alpha=k)
        model_.fit(x_train, y_train)
        y_train_pred = model_.predict(x_train)
        y_test_pred = model_.predict(x_test)
        train_score, test_score = model_.score(x_train, y_train), model_.score(x_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
    index = test_scores.index(max(test_scores))
    best_alpha = index * seq + begin
    if plot:
        plt.plot(krange, test_scores)
        plt.xlabel('Alpha values')
        plt.ylabel('R^2 of Lasso regression')
        plt.title('R^2 with different alpha value using Lasso regression')
        plt.xlim([-end / 10, end])
        plt.show()
        print('Lasso:Best alpha=', best_alpha, ' R^2=', '%.3f' % max(test_scores))
    best_model = api(alpha=best_alpha, l1_ratio=l1_ratio) if model == 'elasticnet' else api(alpha=best_alpha)
    return api(alpha=best_alpha)


def Training(x_train, y_train, model, scale=True, **kwargs):
    if scale:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
    model.fit(x_train, y_train)
    return model


def Testing(fake_testing_x, model, standard=True, **kwargs):
    x_test = fake_testing_x.iloc[:, 1:]
    if standard:
        scaler = StandardScaler()
        x_test_std = scaler.fit_transform(x_test)
    else:
        x_test_std = x_test
    y_test_pred = model.predict(x_test_std)
    return y_test_pred


def TestingFinal(fake_testing_x, path=''):
    # read_data
    cpi, ppi, x, _ = read_data(path)
    x_train = x.iloc[12:, 1:].values

    # inflation by cpi
    # Train by Extra Trees
    inflation_rate_by_cpi = inflation_rate(cpi, 'CPI')
    y_train = inflation_rate_by_cpi['inflation_rate'].values
    model = ExtraTreesRegressor(n_estimators=100, random_state=1)
    model = Training(x_train, y_train, model, scale=False)
    cpi_pred = Testing(fake_testing_x, model, standard=False)

    # inflation by ppi
    # Train by Lasso Regression
    inflation_rate_by_ppi = inflation_rate(ppi, 'PPI')
    y_train = inflation_rate_by_ppi['inflation_rate'].values
    model = Lasso(alpha=0.0002)
    model = Training(x_train, y_train, model)
    ppi_pred = Testing(fake_testing_x, model)

    # outcomes
    outcomes = pd.DataFrame(columns=['CPI', 'PPI'])
    outcomes['CPI'], outcomes['PPI'] = cpi_pred, ppi_pred
    return outcomes


def test():
    real_x = pd.read_csv('real_X.csv', encoding='gb2312')
    cpi, ppi, _, _ = read_data('')
    ans = TestingFinal(real_x, path='')
    real_cpi = pd.read_csv('real_cpi.csv')
    real_ppi = pd.read_csv('real_ppi.csv')
    ans.to_excel('real_testing_ans.xlsx')
    # cpi = pd.concat([cpi, real_cpi], ignore_index=True).iloc[-42:]
    # ppi = pd.concat([ppi, real_ppi], ignore_index=True).iloc[-42:]
    # inflation_rate_by_cpi = inflation_rate(cpi, 'CPI')
    # inflation_rate_by_ppi = inflation_rate(ppi, 'PPI')



if __name__ == '__main__':
    # Load all data
    path = ''
    cpi, ppi, x, fake_testing_x = read_data(path)
    inflation_rate_by_cpi = inflation_rate(cpi, 'CPI')
    inflation_rate_by_ppi = inflation_rate(ppi, 'PPI')

    # Choose predictors
    x_train = x.iloc[12:, 1:].values

    # Inflation rate by cpi
    y_train = inflation_rate_by_cpi['inflation_rate'].values

    # Inflation rate by ppi
    y_train = inflation_rate_by_ppi['inflation_rate'].values

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    # Training
    scaler = StandardScaler()
    x_train_std, x_test_std = scaler.fit_transform(x_train), scaler.fit_transform(x_test)

    # RandomForest Model
    forest = RandomForestRegressor(n_estimators=51, criterion='mse', random_state=1, n_jobs=-1, max_depth=5)
    train_and_evaluate(forest, x_train, x_test, y_train, y_test)

    # Linear Regression
    linear = LinearRegression()
    train_and_evaluate(linear, x_train_std, x_test_std, y_train, y_test)

    # SVM Model
    # rbf
    rbf = svm.SVR(kernel="rbf", C=1)
    train_and_evaluate(rbf, x_train_std, x_test_std, y_train, y_test)

    # poly
    poly = svm.SVR(kernel="poly")
    train_and_evaluate(poly, x_train_std, x_test_std, y_train, y_test)

    # linear
    svm_linear = svm.SVR(kernel="linear")
    train_and_evaluate(svm_linear, x_train_std, x_test_std, y_train, y_test)

    # Adaboosting for RandomForest
    ada = AdaBoostRegressor(base_estimator=forest, n_estimators=100, learning_rate=0.2, random_state=1)
    train_and_evaluate(ada, x_train_std, x_test_std, y_train, y_test)

    # Bagging for Linear Regression
    bag = BaggingRegressor(base_estimator=linear, n_estimators=50, max_samples=75, max_features=100, bootstrap=True,
                           bootstrap_features=False, n_jobs=-1, random_state=1)
    train_and_evaluate(bag, x_train_std, x_test_std, y_train, y_test)

    # SGD regressor
    sgd = SGDRegressor()
    train_and_evaluate(sgd, x_train_std, x_test_std, y_train, y_test)

    # Lasso -- best for ppi
    lasso = RegularizedRegression(x_train_std, x_test_std, y_train, y_test, begin=0.0001, end=0.0003, seq=0.00001)
    train_and_evaluate(lasso, x_train_std, x_test_std, y_train, y_test)

    # Ridge
    ridge = RegularizedRegression(x_train_std, x_test_std, y_train, y_test, model='ridge', begin=0.01, end=2, seq=0.01)
    train_and_evaluate(ridge, x_train_std, x_test_std, y_train, y_test)

    # ElasticNet
    elastic = RegularizedRegression(x_train_std, x_test_std, y_train, y_test, model='elasticnet', begin=0.01, end=2,
                                    seq=0.01)
    train_and_evaluate(elastic, x_train_std, x_test_std, y_train, y_test)

    # Extra Trees -- best for cpi
    extra_trees = ExtraTreesRegressor(n_estimators=100, random_state=1)
    train_and_evaluate(extra_trees, x_train, x_test, y_train, y_test)
