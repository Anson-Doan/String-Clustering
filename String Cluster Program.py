import py_stringclustering as scl
import os
from sklearn.cluster import AgglomerativeClustering

# Get the datasets directory
from py_stringclustering.utils.generic_helper import get_install_path
datasets_dir = get_install_path() + os.sep + 'tests' + os.sep + 'test_datasets'
path_big_ten = datasets_dir + os.sep + 'big_ten.txt'

# Read the strings from file
# inname = input("Enter a file to read from: ")
# outname = input("Enter a file to read to: ")
# nClusters = int(input("How many clusters? "))
#inname = path_big_ten
inname = 'sampleStrings.txt'
outname = 'clusteringOutput.txt'
nClusters = 5
df = scl.read_data(inname)
df.head()

print(len(df))

import py_stringmatching as sm
import py_stringsimjoin as ssj

# Block using Jaccard join with jacc3gr(s1, s2) >= 0.3
# Returns a DataFrame containing pairs of string IDs that satisfy the blocking condition
trigramtok = sm.QgramTokenizer(qval=3)
#blocked_pairs = ssj.jaccard_join(df, df, 'id', 'id', 'name', 'name', trigramtok, 0.3)
blocked_pairs = ssj.overlap_coefficient_join(df, df, 'id', 'id', 'name', 'name', trigramtok, 0.8)
print(blocked_pairs.head())

# Define clustering similarity measure
jaccsim = sm.Jaccard()
jarsim = sm.Jaro()

# Calculate sim scores
# Returns a list of triplets of the form (id1, id2, sim)
#sim_scores = scl.get_sim_scores(df, blocked_pairs, trigramtok, jaccsim)
sim_scores = scl.get_sim_scores(df, blocked_pairs, trigramtok, jarsim)
print (sim_scores[:10])

# Returns a NumPy matrix containing the similarities in sim_scores and zero everywhere else
sim_matrix = scl.get_sim_matrix(df, sim_scores)
print(sim_matrix)

aggcl = AgglomerativeClustering(n_clusters=nClusters, affinity='precomputed', linkage='complete')

### Returns a list of cluster labels
labels = aggcl.fit_predict(sim_matrix)

### Returns a list of clusters where each cluster is a list of strings
str_clusters = scl.get_clusters(df, labels)
fout = open(outname, 'wt')
print (str_clusters)
for category in str_clusters:
    for item in category:
        fout.write(item + '\n')
    fout.write('\n')
fout.close()