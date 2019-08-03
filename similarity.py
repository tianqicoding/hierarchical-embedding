#######################import
import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
from random import sample
import pandas as pd 
import numpy as np 
import csv
import sys
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
t=381
#######################restrict on whitelist

######################load whitelist
tagSet={}
c1=100
c2=0.05
for line in open('may/data/node_'+str(c1)+'_'+str(c2)+'.csv'):
	hashtagName=line.strip('\n').split('\t')[0]
	tagSet[hashtagName]=len(tagSet)

	
l_att=np.zeros((len(tagSet), t))

#####################load hashtags
reader = DataFileReader(open("attributes/raw/attributes_hashtag.avro", "rb"), DatumReader())
print(len(tagSet))

cnt=0
edge={}
a=set(tagSet.keys())
s={}
name=set()
ltag=[]

for item in reader:
	#print(item)
	h=item['keySchema']['entityUrn'].split(':')[-1]
	if h not in tagSet:
		continue
	a.remove(h)
	ltag.append(h)
	l=item['valueSchema']['FeatureVector']
	for i in range(len(l)):
		name.add(l[i]['name'])
		if (l[i]['name'], l[i]['term']) not in s:
			s[(l[i]['name'], l[i]['term'])]=cnt
			cnt+=1
		if l[i]['name']=='langsCompatibleWithHashtagChars':
			l_att[tagSet[h], s[(l[i]['name'], l[i]['term'])]]=1
		else:
			l_att[tagSet[h], s[(l[i]['name'], l[i]['term'])]]=l[i]['value']
	#l_att.append(ch)
print(len(a))
df=pd.DataFrame(l_att)
df.to_csv('attributes/'+str(c1)+'_'+str(c2)+'.csv')
df_nodes=pd.DataFrame({'hashtags':list(a)})
df_nodes.to_csv('attributes/remain_'+str(c1)+'_'+str(c2)+'.csv', index=False, header=None)



# print(len(l_att))
# A=np.array(l_att)
# A_sparse = sparse.csr_matrix(A)
# print('computed matrix')
# similarities = cosine_similarity(A_sparse)
# print('computed similarity')
# np.savetxt("similarity.csv", similarities, delimiter=",")


# df_nodes=pd.DataFrame({'hashtags':ltag})
# df_nodes.to_csv('snode_100_0.1.csv', index=False, header=None)


# attr=[]


# data={'hashtag1':u, 'hasgtag2':v, 'attr':attr}
# df=pd.DataFrame(data)
# df.to_csv('attr_100_0.7.csv', index=False)
# df_nodes=pd.DataFrame({'hashtags':list(a)})
# df_nodes.to_csv('remain_100_0.7.csv', index=False, header=None)
# print('done nodes')
# df_nodes=pd.read_csv('snode_100_0.1.csv', header=None, names=[1])[1].tolist()
# print(len(df_nodes),'loaded nodes')
# similarity=np.loadtxt("similarity.csv", delimiter=",")
# print(similarity.shape, 'loaded similarities')

# ltag=[]
# for t in df_nodes:
# 	ltag.append(t)

# u=[]
# v=[]
# attr=[]
# for i in range(similarity.shape[0]):
# 	for j in range(i+1, similarity.shape[0]):
# 		if similarity[i, j]>0:
# 			u.append(tagSet[ltag[i]])
# 			v.append(tagSet[ltag[j]])
# 			attr.append(similarity[i, j])
# 	if i%1000==0:
# 		print(i)
	
# data={'hashtag1':u, 'hasgtag2':v, 'attr':attr}
# df=pd.DataFrame(data)
# print('done edge')
# df.to_csv('attr_100_0.1.csv', index=False)
# # df_nodes=pd.DataFrame({'hashtags':list(a)})
# # df_nodes.to_csv('remain_100_0.7.csv', index=False, header=None)
# # print(len(tagSet), len(s), len(a))
