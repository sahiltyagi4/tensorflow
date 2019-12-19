import seaborn as sb
import matplotlib.pyplot as plt
import csv

iterationtimes = []
with open('/Users/sahiltyagi/Desktop/periteration/aspmodgradvarbatch/tf-worker-1/worker1.csv') as csvfile:
	f = csv.reader(csvfile)
	for line in f:
		iterationtimes.append(float(line[0]))

csvfile.close()

ax = sb.distplot(iterationtimes)
plt.show()