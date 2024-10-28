import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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


# from sklearn.model_selection import cross_val_score

# for k in range(3,20,2):
#     model = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(model,X_train,y_train)
#     print(k,scores.mean())
#算出k=13是最拟合的


# 训练k-近邻模型
knn = KNeighborsClassifier(n_neighbors=13)  # 可以调整n_neighbors的值
knn.fit(X_train, y_train)

# 使用测试集测试模型
y_pred = knn.predict(X_test)
acc_score = accuracy_score(y_test,y_pred)
print(acc_score)
#约为0.91

# 评估模型，计算F1值
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1}')
#约为0.86