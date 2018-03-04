import scipy.optimize as opt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import matplotlib.pyplot as plt
# import tensorflow as tf
from sklearn.metrics import classification_report#这个包是评价报告


def get_X(df):#读取特征
#     """
#     use concat to add intersect feature to avoid side effect
#     not efficient for big dataset though
#     """
    ones = pd.DataFrame({'ones': np.ones(len(df))})#ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return data.iloc[:, :-1].as_matrix()  # 这个操作返回 ndarray,不是矩阵


def get_y(df):#读取标签
#     '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])#df.iloc[:, -1]是指df的最后一列


def normalize_feature(df):
#     """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())#特征缩放


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))
# X @ theta与X.dot(theta)等价


def gradient(theta, X, y):
#     '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)


if __name__ == '__main__':

	data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
	data.head()#看前五行
	print(data.head(), data, data.describe())


	sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))
	sns.lmplot('exam1', 'exam2', hue='admitted', data=data, 
	           size=6, 
	           fit_reg=False, 
	           scatter_kws={"s": 50}
	          )
	plt.show()#看下数据的样子


	X = get_X(data)
	y = get_y(data)
	print(X.shape)
	print(y.shape)
	# (100, 3)
	# (100,)

	fig, ax = plt.subplots(figsize=(8, 6))
	ax.plot(np.arange(-10, 10, step=0.01),
	        sigmoid(np.arange(-10, 10, step=0.01)))
	ax.set_ylim((-0.1,1.1))
	ax.set_xlabel('z', fontsize=18)
	ax.set_ylabel('g(z)', fontsize=18)
	ax.set_title('sigmoid function', fontsize=18)
	plt.show()

	theta = theta=np.zeros(3) # X(m*n) so theta is n*1

	cost(theta, X, y)
	gradient(theta, X, y)

	res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
	print(res)


	final_theta = res.x
	y_pred = predict(X, final_theta)
	print(classification_report(y, y_pred))

	print(res.x) # this is final theta

	coef = -(res.x / res.x[2])  # find the equation
	print(coef)
	x = np.arange(130, step=0.1)
	y = coef[0] + coef[1]*x

	data.describe()  # find the range of x and y


	sns.set(context="notebook", style="ticks", font_scale=1.5)
	sns.lmplot('exam1', 'exam2', hue='admitted', data=data, 
	           size=6, 
	           fit_reg=False, 
	           scatter_kws={"s": 25}
	          )
	plt.plot(x, y, 'grey')
	plt.xlim(0, 130)
	plt.ylim(0, 130)
	plt.title('Decision Boundary')
	plt.show()