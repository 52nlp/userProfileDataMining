import sys
from time import time
from sklearn import preprocessing, linear_model
from scipy.sparse import bmat
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer,CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
# from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# 从整个训练集数据集中抽取部分数据作为训练模型的训练集数据和测试集数据，并且指定要使用的目标变量
def input_data(train_file,divide_number,end_number,tags):
    train_words = []
    train_tags=[]
    test_words = []
    test_tags=[]
    with open(train_file, 'r',encoding='gb18030') as f:
        text=f.readlines()

        # 构建训练集数据
        train_data=text[:divide_number]   
        for single_query in train_data:
        	# 先将所有的字段分割
            single_query_list = single_query.split(' ')
            # 去除 ID 字段
            single_query_list.pop(0)#id
            # 标签确定的情况下构建样本
            if(single_query_list[tags]!='0'):
            	# 构建训练集样本的目标变量
                train_tags.append(single_query_list[tags])
                # 删除3个目标变量，剩下关键词
                single_query_list.pop(0)
                single_query_list.pop(0)
                single_query_list.pop(0)
                # 列表转换为字符串，列表中的逗号转换为空格，将字符串所有的单引号去掉，将列表的左中括号和右中括号去掉，最后将换行符去掉
                # 所以最后剩下的是所有关键词构成的字符串，它们分别用逗号分隔
                train_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))

        #构建测试集数据，构建的方法和训练集数据的构建是一样的
        test_data=text[divide_number:end_number]   
        for single_query in test_data:
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)#id
            if(single_query_list[tags]!='0'):
                test_tags.append(single_query_list[tags])
                single_query_list.pop(0)
                single_query_list.pop(0)
                single_query_list.pop(0)
                test_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))

    print('input_data done!')

    # 返回构建的训练集输入，训练集目标变量，测试集输入，测试集目标变量
    # 将训练集数据和测试集数据的目标变量进行 onehot 编码
    # 由于它们都是字符串，并且是从1开始的，所以要先用 LabelEncoder 进行处理。
    le = LabelEncoder()
    train_tags = le.fit_transform(train_tags)
    oe = OneHotEncoder()
    train_tags = oe.fit_transform(train_tags.reshape(-1,1)).toarray()

    le = LabelEncoder()
    test_tags = le.fit_transform(test_tags)
    oe = OneHotEncoder()
    test_tags = oe.fit_transform(test_tags.reshape(-1,1)).toarray()


    return train_words, train_tags, test_words, test_tags

    
# 将训练集数据和测试集数据转换为 tf-idf 特征矩阵
def tfidf_vectorize(train_words, train_tags, test_words):
    print ('*************************\nTfidfVectorizer\n*************************')   
    # sublinear_tf 用于对 tf 进行缩放，例如把 tf 替换成 1 + log(tf)
    # 一般情况下我们是这么做的，而不是直接使用 tf（也就是关键词在文档中出现的次数）
    # 其实可以在这里设置 stop_words，而不用在前面分词的时候处理
    # 现在的问题是如何设置中文的停用词
    tv = TfidfVectorizer(sublinear_tf = True)
                                          
    tfidf_train_2 = tv.fit_transform(train_words);
    tfidf_test_2 = tv.transform(test_words)
    print ("the shape of train is "+repr(tfidf_train_2.shape))  
    print ("the shape of test is "+repr(tfidf_test_2.shape))
    # train_data,test_data=feature_selection_chi2(tfidf_train_2,train_tags,tfidf_test_2,n_dimensionality) 
    return  tfidf_train_2, tfidf_test_2


    
# 使用简单的神经网络来进行分类
# 先用训练集数据训练模型，然后对测试集数据进行预测
def nn_single(train_data,test_data,train_tags, test_tags, tags): 
    print ('******************************神经网络*****************************' )

    if tags == 0:
        classes = 6
    elif tags == 1:
        classes = 2
    elif tags == 2:
        classes = 6
    else:
        print('标签不合规！')
        exit(0)

    # train_tags = np_utils.to_categorical(train_tags, num_classes=classes)
    # test_tags = np_utils.to_categorical(test_tags, num_classes=classes)


    # 构建模型
    # 模型存放路径
    model_path = 'nn_models/best_model_tags'+str(tags)+'.h5'
    model_file = Path(model_path)
    # 检测是否有已经训练好的模型，如果有就加载继续训练，如果没有就创建新模型
    if  model_file.is_file():
        print('loading model'.center(60, '*'))
        model = load_model(model_path)
        print('model loaded'.center(60, '*'))
    else:
        print('creating model'.center(60, '*'))
        model = Sequential([
            Dense(32, input_dim=np.shape(train_data)[1]),
            Activation('relu'),
            Dense(classes),
            Activation('softmax')
        ])
        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        print('model created'.center(60, '*'))

    # 训练模型
    # 创建回调函数
    checkpoint = ModelCheckpoint(model_path,monitor='val_loss',save_best_only=True, verbose=1)
    callbacks_list = [checkpoint]
    train_result = model.fit(train_data, train_tags, 
                    epochs=10, 
                    batch_size=2048, 
                    validation_split=0.11, 
                    callbacks = callbacks_list,
                    verbose=1)

    # 评估最优模型
    model = load_model(model_path)
    loss, accuracy = model.evaluate(test_data, test_tags)
    return loss, accuracy



# 分别对3个目标变量进行测试
def test():
    # 0 对应 age
    # 1 对应 Gender
    # 2 对应 Education
    # 我们分别以这三个标签作为我们的目标变量进行训练
    test_single(0)
    test_single(1)
    test_single(2)


# 对指定的目标变量（age，Gender，Education）进行测试
def test_single(tags):
    train_file = 'train_data_fenci.txt'
    # 测试集起始样本位置
    divide_number=15500
    # 测试集终止样本位置
    end_number=17633

    print('file:'+train_file)
    print('tags:%d   ' % tags )
    #将数据分为训练与测试，获取训练与测试数据的标签
    train_words, train_tags, test_words, test_tags = input_data(train_file,divide_number,end_number,tags)
    # 使用TFIDF将关键词转换为特征矩阵
    train_data,test_data= tfidf_vectorize(train_words, train_tags, test_words)
    
    loss, accuracy =nn_single(train_data,test_data,train_tags, test_tags, tags)
    print("accuracy score: ", accuracy)
    print("loss: ", loss)



def main():
    # 如果第一个参数是 test，那么对3个目标变量分别进行测试，看看分类效果如何
    if(sys.argv[1]=="test"):
        test()

if __name__ == '__main__':
    main()
