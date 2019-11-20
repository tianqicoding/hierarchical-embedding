import pandas as pd
import csv
reader = pd.read_csv('mammal_closure.csv')

#reader.to_csv('mammal_closure.tsv', sep='\t', header=False)
mammal_name={}
mammal_namelist=[]
def update(row):
	#print(row)
	if row['id1'] not in mammal_name:
		 mammal_name[row['id1']]=len(mammal_name)
		 mammal_namelist.append(row['id1'])
	if row['id2'] not in mammal_name:
		 mammal_name[row['id2']]=len(mammal_name)
		 mammal_namelist.append(row['id2'])
	row['id1']=mammal_name[row['id1']]
	row['id2']=mammal_name[row['id2']]
	#print(row)
	return row
reader=reader.apply(update, axis=1)
print(reader.index)
reader.to_csv('mammal_closure.tsv', sep='\t', header=False, index=False)
with open('mammal_name.csv', 'w') as writeFile:
	writer = csv.writer(writeFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(mammal_namelist)):
		writer.writerow([mammal_namelist[i]])
