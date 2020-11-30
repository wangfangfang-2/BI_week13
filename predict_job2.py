import pandas as pd 
from collections import defaultdict
import networkx as nx 
import matplotlib.pyplot as plt 
import random
#设置黑体、正常显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


#数据加载
file = 'jobs_4k.csv'
content = pd.read_csv(file)
#print(content)
position_names = content['positionName'].tolist()
skill_lables = content['skillLables'].tolist()


skill_position_graph = defaultdict(list)
for p, s in zip(position_names, skill_lables):
    #print(s)
    skill_position_graph[p] += eval(s)
#print(skill_position_graph)

G=nx.Graph(skill_position_graph)
#以20个随机选择的工作岗位为例
sample_nodes = random.sample(position_names,k=20)
#print(sample_nodes)
#做一张图 初始化节点（刚开始为随机的20个职位）
sample_nodes_connections = sample_nodes
#给随机的20个职位，添加相关的技能
for p,skills in skill_position_graph.items():
    if p in sample_nodes:
        sample_nodes_connections += skill_position_graph
        #抽取G中的子图
sample_graph= G.subgraph(sample_nodes_connections)
plt.figure(figsize=(50,30))
pos = nx.spring_layout(sample_graph,k=1)
nx.draw(sample_graph,pos,with_labels=True,node_size=30,font_size=10)
#plt.show()
#使用PageRank算法，对核心能力和核心职位进行影响力排序
pr = nx.pagerank(G,alpha=0.9)
ranked_position_and_ability = sorted([(name,value) for name,value in pr.items()],key= lambda x:x[1],reverse=True )
#print(ranked_position_and_ability)

#特征X，去掉salary字段
X_content= content.drop(['salary'],axis=1)
#目标Target
target = content['salary'].tolist()

#将X_content内容拼接成字符串，设置为merged字段
X_content['merged'] = X_content.apply(lambda x:''.join(str(x)),axis=1)
#print(X_content['merged'][0])
#转化为list
X_string = X_content['merged'].tolist()

import jieba
import re
def get_one_row_job_string(x_string_row):
    job_string = ''
    for i,element in enumerate(x_string_row.split('\n')):
        if len(element.split()) ==2:
            _,value = element.split()
            #i=0 为id字段，需要去掉
            if i==0:
                continue
            #只保存value
            job_string +=value
    return job_string
def token(string):
    return re.findall('\w+',string)

cutted_X=[]    
for i ,row in enumerate(X_string):
    #print(row)
    job_string = get_one_row_job_string(row)
    cutted_X.append(' '.join(list( jieba.cut(' '.join(token(job_string))))))
print(cutted_X)
#提取文本特征 使用tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cutted_X)
#print(X)
#print(target[:10])

import numpy as np 
target_numical = [np.mean(list(map(float,re.findall('\d+',s)))) for s in target]
print(target_numical)
Y=target_numical
#使用KNN算法预测
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=2)
model.fit(X,Y)
def predict_by_label(test_string,model):
    #分词
    test_words= list(jieba.cut(test_string))
    #转换为tfidf向量
    test_vec = vectorizer.transform(test_words)
    #预测
    y_pred = model.predict(test_vec)
    return y_pred[0]
test = '测试 北京 3年 专科'
print(test,predict_by_label(test,model))

test2 = '测试 北京 4年 专科'
print(test2,predict_by_label(test2,model))
test3 = '算法 北京 4年 本科'
print(test3,predict_by_label(test3,model))
test4 = 'UI 北京 4年 本科'
print(test4,predict_by_label(test4,model))
persons = ["广州Java本科3年掌握大数据",
"沈阳Java硕士3年掌握大数据", 
"沈阳Java本科3年掌握大数据", 
"北京算法硕士3年掌握图像识别"
]
for p in persons:
    print("{}的薪资预测为{}".format(p,predict_by_label(p,model)))
