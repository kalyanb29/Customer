import pandas as pd
import os
import numpy as np
from time import gmtime, strftime

test_data = pd.read_csv(os.getcwd() + '/test/all_features.csv', sep=' ')
test_ids = test_data['id'].values
prob = np.loadtxt(os.getcwd() + '/probabilities.txt')

of = open(os.getcwd() + '/submission_'+ strftime("%d-%m_%H.%M.%S",gmtime()) +".csv","w")
of.write("id,repeatProbability\n")
for i in range(len(test_ids)):
	of.write( "%d,%.12f\n" % (test_ids[i], prob[i]) )
