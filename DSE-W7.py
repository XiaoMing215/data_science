import pandas as pd

# 读取数据集
data = pd.read_csv(r"D:\布布的文档\大二上\数据科学导论\第七次\bike.csv")
print("numbers before clean:",len(data))
# 剔除id列
data = data.drop('id', axis=1)
# 筛选上海市的数据并剔除city列
data = data[data['city'] == 1].drop('city', axis=1)
# 将hour列中6点-18点统一为1，19点-次日5点统一为0
data['hour'] = data['hour'].apply(lambda x: 1 if 6 <= x <= 18 else 0)
# 提取y列并转换为numpy列向量，然后剔除y列
y = data['y'].values
data = data.drop('y', axis=1)
# 将DataFrame对象转换为Numpy数组
data = data.values

print("numbers after clean:",len(data))

from sklearn.model_selection import train_test_split

# 划分训练集和测试集
train_data, test_data, train_y, test_y = train_test_split(data, y, test_size=0.2, random_state=42,shuffle=True)

from sklearn.preprocessing import MinMaxScaler
# 初始化归一化处理对象
scaler = MinMaxScaler()

# 对训练集数据、测试集数据进行归一化
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# 对训练集标签、测试集标签进行归一化
train_y = scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
test_y = scaler.transform(test_y.reshape(-1, 1)).flatten()

from sklearn.linear_model import LinearRegression

# 构建线性回归模型
model = LinearRegression()
model.fit(train_data, train_y)

# 利用测试集对模型进行评估
predictions = model.predict(test_data)

print("My pridiction:",predictions)

# from sklearn.metrics import mean_squared_error

# # 计算均方根误差(RMSE)
# mse = mean_squared_error(test_y, predictions)
# rmse = mse ** 0.5
# print(f"RMSE: {rmse}")


from sklearn.metrics import root_mean_squared_error

# 计算均方根误差(RMSE)
rmse = root_mean_squared_error(test_y, predictions)
print(f"RMSE: {rmse}")


