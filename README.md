# hierarchical-embedding
hierarchical-embedding
embed.py is the file to embed. After that using get_hierarchy.py to get the hierarchical structure, and evaluate_reconstruction.py to evaluate. Codes in folder hie are modified from heat(https://github.com/DavidMcDonald1993/heat).

embed.py: 
	input: 
		--edgelist: .tsv file, each line as u, v, w as the edge (u, v) with weight w.
		-e: number of epochs, default to 10
		-d: embedding dimension, usually 5
		--walks: the path to save random walks
		--embedding: the path to save embeddings
		--directed: falg of input directed graph
		--no-walks: flag not to use random walk

evaluate_reconstruction.py:
	input:
		--edgelist
		--directed
		--embedding: embedding file name
		--test-results-dir: path to save results

get_hierarchy.py:
	input:
		--nodelist: node list
		--edgelist: edge list, .tsv file
		--embedding: embedding name
		--hierarchyfile: file to save hierarchical structure
		--directed