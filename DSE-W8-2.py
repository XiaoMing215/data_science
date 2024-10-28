import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
# 加载数据
df = pd.read_csv('数据科学最后一次作业/fraudulent.csv')

# 数据预处理
# 处理缺失值，这里我们选择用每列的众数来填充缺失值
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 分离特征和标签
X = df_imputed.drop('y', axis=1)
y = df_imputed['y']

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# 初始化决策树分类器
# dt = DecisionTreeClassifier(
#     max_depth=11,  # 最大深度
#     min_samples_split=5,  # 分裂内部节点所需的最小样本数
#     min_samples_leaf=1,  # 叶子节点所需的最小样本数
#     random_state=1  # 随机种子，确保结果可复现
# )

# 定义要搜索的超参数网格
# param_grid = {
#     'max_depth': [3,8,9,10,11,12],
#     'min_samples_split': [2,3,5],
#     'min_samples_leaf': [1, 2, 4]
# }

# # 创建网格搜索对象
# grid_search = GridSearchCV(dt, param_grid, cv=5)

# # 进行网格搜索
# grid_search.fit(X_train, y_train)

# # 输出最佳参数组合
# print(grid_search.best_params_)
# #组合是：{'max_depth': 11, 'min_samples_leaf': 1, 'min_samples_split': 5}

# 训练决策树模型
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, y_train)

# 使用测试集测试模型
y_pred = dt.predict(X_test)
# 使用测试集测试模型
acc_score = accuracy_score(y_test,y_pred)
print(acc_score)
#约为0.91
# 评估模型，计算F1值
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1}')