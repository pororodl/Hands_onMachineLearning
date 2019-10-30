from __future__ import division,print_function,unicode_literals

import os
import sys
import numpy as np
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import hashlib


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT+HOUSING_PATH+'/housing.tgz'

def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    # 如果数据还没有下载下来，就创建这文件，把下载的数据放在这个文件夹下
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    # 定义下载下来的压缩包的名字
    tgz_path = os.path.join(housing_path,'housing_tgz')

    # 用来对服务器发送请求来获取数据housing_url是下载链接，tgz_path是下载下来之后的压缩包路径和名字（加上了文件路径）
    urllib.request.urlretrieve(housing_url,tgz_path)

    # 解压文件
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)  # 解压这个文件夹下的所有文件
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,'housing.csv')
    data = pd.read_csv(csv_path)
    return data



def split_train_test(data,test_ratio):
    np.random.seed(42)      # 加上种子之后如果整个数据集不变换的话，多次生成的随机序列就是一样的，但是如果后续有加入数据的话，进入测试集的样本就会发生变化。
    shuffled_indices = np.random.permutation(len(data))  # 随机产生0~len(data)-1的整数，无序，但是会每次生成的不一样
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

def test_set_check(identifier,test_ratio,hash):
    print('hash',hash(np.int64(identifier)).digest()[-1])
    return hash(np.int64(identifier)).digest()[-1]<256*test_ratio

def split_train_test_by_id(data,test_ratio,id_column,hash=hashlib.md5):
    ids = data[id_column]
    in_test_set= ids.apply(lambda id_:test_set_check(id_,test_ratio,hash))  # ？？？？？这个函数的用法需要记录
    return data.loc[~in_test_set],data.loc[in_test_set]

if __name__ == '__main__':
    data = load_housing_data()
    pd.set_option('display.max_columns',None)   # set_option 可以设置显示的一些格式
    # print(data.head()) # 查看前5行
    # print(data.info()) # 查看每一列的数量和类型，可以查看是否有缺失值
    # print(data['ocean_proximity'].value_counts())  # 查看这一个属性有多少种值和每种值分别有多少个样本
    # print(data.describe()) # 查看每一种属性的值的分布情况 （平均值，标准差，分位数）

    # 绘制每个数值属性的直方图
    # data.hist(bins=50,figsize=(20,15))  # bins 是箱数
    # plt.show()

    # train_set ,test_set = split_train_test(data,test_ratio=0.2)
    # print('train:',len(train_set),'test:',len(test_set))
    # print('train_set[9]\n',train_set.iloc[9])
    # print('test_set[10]\n',test_set.iloc[10])

    # 解决上面所述的问题的方法是给每个样本一个唯一的标识符来决定是否进入测试集。
    # 可以计算每个实例标识符的hash值，取hash值的最后一个字节，如果该值小于等于51（大概是256*20%），就放入测试集
    # 如果新加入了数据的话，原来在测试集里的仍然在测试集里面

    # 采用列索引作为唯一标识符，但是要保证如果加入数据时，在数据集后面加入数据，保证前面的数据的列索引值不变
    # housing_with_id = data.reset_index()  # 给data_frame加上列索引
    # train_set, test_set = split_train_test_by_id(housing_with_id,test_ratio=0.2,id_column='index')   # 注意这里传入参数时，如果test_ratio用了test_ratio=0.2,那么后面的一定要加上参数名id_column=

    # 可以采用数据中的经纬度组合成id
    # housing_with_id['id']= data['longitude']*1000+data['latitude']
    # train_set, test_set = split_train_test_by_id(housing_with_id,0.2,'id')  # 按照顺序传入就不需要


    # 用sklearn 库可以一条语句解决
    # train_test_split(data,test_size=0.2,random_state=42)  # 优点在于可以同时划分行数相同的几个数据集，如果特征在一个dataframe 而标签另外在一个dataframe，那么就很容易切分了

    # 需要根据某一个特征进行分层采样时，如果这个特征值是连续的数值，就需要对其进行离散化
    data['income_cat'] = np.ceil(data['median_income']/1.5)   # ceil是向上取整  floor是向下取整，around是四舍五入
    data['income_cat'].where(data['income_cat']<5,5.0,inplace=True)  # 把大于5的替换为5
    # print(data['income_cat'].value_counts())

    # 分层采样
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(data,data['income_cat']): # 根据income_cat的每种类别的数量进行采样，做到分层采样的目的
        # print('train_index',train_index,'test_index',test_index)
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]  # 这里为什么用loc 不用iloc,两者的区别
    # print('train',len(strat_train_set))
    # print('test',len(strat_test_set))
    # print(data['income_cat'].value_counts()/len(data))
    # print(strat_train_set['income_cat'].value_counts()/len(strat_train_set))
    # print(strat_test_set['income_cat'].value_counts()/len(strat_test_set))

    # 删除掉income_cat 列
    for set in (strat_train_set,strat_test_set):
        set.drop(['income_cat'],axis=1,inplace = True) # inplce 是否在原对象基础上进行修改

    # 建立一个训练集的副本来对数据进行可视化，可以不损害训练集
    housing = strat_train_set.copy()
    # print(type(housing))       # dataFrame可以直接进行可视化
    # housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.1)  # 设置透明度，可以看清高密度数据点的位置alpha值为透明度值，值越大越不透明 可以对点的密集地方显示的更深
    # 用点的大小表示人口的数量（参数s),用颜色代表价格(参数c),jet预定义颜色表，参数cmap
    # housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,s=housing['population']/100,label='population',c='median_house_value',cmap=plt.get_cmap('jet'),colorbar=True,)
    # plt.legend()
    # plt.show()

    # 查看相关性
    # 由于特征不是很多，所以可以看看每个属性和房屋中位数的相关性分别是多少
    # corr_matrix = housing.corr()
    # print(corr_matrix)
    # corr_house_value = corr_matrix['median_house_value'].sort_values(ascending=False)
    # print(corr_house_value)

    # 另一种查看相关性的方法
    # from pandas.plotting import scatter_matrix
    #
    # attributes = ['median_house_value','median_income','total_rooms','housing_median_age']
    # scatter_matrix(housing[attributes],figsize=(12,8))
    # plt.show()

    # housing.plot(kind='scatter',x='median_income',y='median_house_value',alpha=0.1)
    # plt.show()

    # 创建一些更有意义的新属性
    housing['rooms_per_household']=housing['total_rooms']/housing['households']
    housing['bedrooms_per_room']=housing['total_bedrooms']/housing['total_rooms']
    housing['population_per_household']=housing['population']/housing['households']

    corr_matrix = housing.corr()
    # print(corr_matrix)
    corr_house_value = corr_matrix['median_house_value'].sort_values(ascending=False)
    print(corr_house_value)

    # 缺失值的处理
    # option1:放弃这些有缺失的数据
    # option2:放弃这个属性
    # option3:将缺失的值填入

