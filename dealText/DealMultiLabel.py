from sklearn.model_selection import train_test_split
import os
import tokenization
import pandas as pd

"""
多标签转换成单标签
"""
import text_deal as dt

def MergeLabel(fileName):
    # 定义自定义分类任务
    classifiaction = {}
    # 读取分类文件
    f = open('data/分类.txt', 'r', encoding='utf-8')
    # 处理分类类别转成字典
    for line in f.readlines():
        key, value = line.replace("\'", "").replace("\n", "").replace(" ", "").split(":")
        classifiaction[key] = value
    # 读取数据源
    dataSet = pd.read_csv(fileName)
    # 数据源长度
    size = len(dataSet.values)
    # 定义返回结果
    returnDataSet = []
    # 处理数据
    for index, test in enumerate(dataSet.values):
        singleT = []
        singleT.append(test[0])
        d = test[1].replace("\'", "").split(",")
        d.sort()
        s = ",".join(d)
        label = classifiaction.get(s)
        singleT.append(label)
        returnDataSet.append(singleT)
    file_data = pd.DataFrame(returnDataSet, columns=['text', 'label'])
    file_data.to_csv("./data/多分类转单分类数据.csv", index=False)


"""抽取测试集"""


def BuildTestSet():
    singleSet = pd.read_csv('data/单标签.csv')
    mutilSet = pd.read_csv('data/多分类转单分类数据.csv')
    # singleSetValue = singleSet.values
    # mutilSetValue = mutilSet.values
    # totalValue = np.vstack((mutilSetValue, singleSetValue))
    totalValueSet = pd.concat([singleSet, mutilSet])
    # 划分数据集
    x = totalValueSet.iloc[:, 0]
    y = totalValueSet.iloc[:, 1]
    # 使用train_test_split函数划分数据集(训练集占75%，测试集占25%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    pd.DataFrame(x_train).to_csv("./data/x_train.csv")
    pd.DataFrame(x_test).to_csv("./data/x_test.csv")
    pd.DataFrame(y_train).to_csv("./data/y_train.csv")
    pd.DataFrame(y_test).to_csv("./data/y_test.csv")
    # return   x_train, x_test, y_train, y_test


def getData(data_dir):
    x_train_path = os.path.join(data_dir, 'x_train.csv')
    y_train_path = os.path.join(data_dir, 'y_train.csv')
    x_train_df = pd.read_csv(x_train_path, encoding='utf-8')
    y_train_df = pd.read_csv(y_train_path, encoding='utf-8')
    train_data = []
    for index, test in enumerate(zip(x_train_df.values, y_train_df.values)):
        guid = 'train-%d' % index  # 参数guid是用来区分每个example的
        daa = test[0][1]
        text_a = tokenization.convert_to_unicode(dt.deal_content(str(test[0][1])))  # 要分类的文本
        label = str(test[1][1])  # 文本对应的情感类别
        # train_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))  # 加入到InputExample列表中
        index += 1
        train_data.append([text_a,label])
    return train_data


if __name__ == '__main__':
    # MergeLabel('./data/多标签.csv')
    # BuildTestSet()

    getData('./data')
