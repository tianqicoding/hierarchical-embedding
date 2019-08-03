
for i in 100 50 20
do
	for j in 0.1 0.05 0.07
	do
		edgelist=may/data/edge_${i}_${j}.tsv
		nodelist=may/data/node_${i}_${j}.csv
		python3 get_hierarchy.py --edgelist ${edgelist} --embedding embeddings/${i}/${j}/alpha=0.00/seed=000/dim=005/00010_embedding.csv --nodelist ${nodelist} --hierarchyfile hierarchy/${i}_${j}.txt --directed
	done

done


