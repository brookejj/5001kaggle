from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import feature_selector
import pandas as pd

df = pd.read_csv('train.csv',index_col = 'id')

df = pd.get_dummies(df)

feature_cols = [col for col in df.columns if col not in ['time']]
X = df[feature_cols]
y = df['time']

# 特征工程
# 1. 把n_jobs里的-1换成16
X['n_jobs'].replace(-1,16,inplace=True)
# 2. 创造新变量C：为'max_iter','n_samples','n_features'的乘积
X['C'] = X.apply(lambda row: row['max_iter'] * row['n_samples']*row['n_features'], axis=1)
# 3. 创造新变量D: 为'n_jobs'的平方
X['D'] = X.apply(lambda row:np.square(row['n_jobs']),axis=1)
X.drop('n_jobs',axis = 1,inplace=True)

# 观察特征重要性

# fs = feature_selector.FeatureSelector(data=X, labels=y)
# fs.identify_collinear(correlation_threshold=0.9)
# fs.identify_zero_importance(task='regression', eval_metric='auc', n_iterations=10, early_stopping=True)
#
# zero_importance_features = fs.ops['zero_importance']
# fs.plot_feature_importances(threshold=0.99, plot_n=12)
# fs.identify_low_importance(cumulative_importance=0.99)

# X_1中为最后选取的特征
X_1 = X[['n_classes', 'penalty_elasticnet', 'penalty_l1',
       'penalty_l2', 'penalty_none', 'C', 'D']]


# ====================== xgboost预测 ========================

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def xgb_grid_search(X,y,cv_params,other_params):
    model = xgb.XGBRegressor(**other_params)
    gsearch = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=30, verbose=1, n_jobs=4)
    gsearch.fit(X, y)
    evalute_result = gsearch.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(gsearch.best_params_))
    print('最佳模型得分:{0}'.format(gsearch.best_score_))

# cv_params = {'n_estimators': [300,400,500,600,700,800]}
# other_params = {'learning_rate': 0.1, 'n_estimators': 900, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
#                     'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
# xgb_grid_search(X_1,y,cv_params,other_params)

# cv_params = {'max_depth': [3, 4, 5, 6], 'min_child_weight': [1, 2, 3, 4]}
# other_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
#                     'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
# xgb_grid_search(X_1,y,cv_params,other_params)

# xgboost超参数
params = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 5, 'min_child_weight': 2, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 6, 'reg_lambda': 3}


model = xgb.XGBRegressor(**params)
model.fit(X_1,y)

# cv_error = cross_val_score(model,X_1,y,cv=30,scoring='neg_mean_squared_error').mean()
# print(cv_error)

x_test = pd.read_csv('test.csv',index_col = 'id')
x_test = pd.get_dummies(x_test)
x_test['n_jobs'].replace(-1, 16, inplace=True)
x_test['C'] = x_test.apply(lambda row: row['max_iter'] * row['n_samples']*row['n_features'], axis=1)
x_test['D'] = x_test.apply(lambda row: np.square(row['n_jobs']), axis=1)
x_test = x_test[['n_classes', 'penalty_elasticnet', 'penalty_l1', 'penalty_l2', 'penalty_none', 'C', 'D']]

result1 = model.predict(x_test)
from pandas import DataFrame as DF
DF(result1).to_csv('result1.csv', index_label='Id', header=['time'])

# ================== catboost预测 =======================
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

xx = X_1.copy()
X_train, X_test, y_train, y_test = train_test_split(xx, y, test_size=0.1, random_state=40)

params = {'depth': 6,
          'iterations': 600,
          'learning_rate': 0.1,
          'l2_leaf_reg': 3,
          'border_count': 32,
          'thread_count': 4,
          'loss_function': 'RMSE',
          'leaf_estimation_method': 'Gradient'
          }

model = CatBoostRegressor(**params)
model.fit(xx, y)

# cv_error = cross_val_score(model,xx,y,cv=2,scoring='neg_mean_squared_error').mean()
# print(cv_error)

x_test = pd.read_csv('test.csv',index_col = 'id')
x_test = pd.get_dummies(x_test)
x_test['n_jobs'].replace(-1,16,inplace=True)
x_test['C'] = x_test.apply(lambda row: row['max_iter'] * row['n_samples']*row['n_features'], axis=1)
x_test['D'] = x_test.apply(lambda row: np.square(row['n_jobs']),axis=1)
x_pre = x_test[['n_classes','penalty_elasticnet', 'penalty_l1', 'penalty_l2','penalty_none', 'C', 'D']]
result2 = model.predict(x_pre)
DF(result2).to_csv('result2.csv', index_label='Id', header=['time'])

# ================= Average ensemble ===================

ave_result = (result1 + result2)/2
DF(ave_result).to_csv('result.csv', index_label='Id', header=['time'])

