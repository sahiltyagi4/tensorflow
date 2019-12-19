import seaborn as sb
import matplotlib.pyplot as plt
import csv

exp_dir = '/Users/sahiltyagi/Desktop/backupscripts/periteration/asporggradstaticbatch/'
master0 = []
with open(exp_dir + 'tf-master-0/master0.csv') as csvfile:
	f = csv.reader(csvfile)
	for line in f:
		if float(line[0]) < 10:
			master0.append(float(line[0]))

csvfile.close()

worker0 = []
with open(exp_dir + 'tf-worker-0/worker0.csv') as csvfile:
	f = csv.reader(csvfile)
	for line in f:
		if float(line[0]) < 10:
			worker0.append(float(line[0]))

csvfile.close()

worker1 = []
with open(exp_dir + 'tf-worker-1/worker1.csv') as csvfile:
	f = csv.reader(csvfile)
	for line in f:
		if float(line[0]) < 10:
			worker1.append(float(line[0]))

csvfile.close()

ax1=sb.distplot(master0, hist=False, rug=False, color='r', label='0.26x cluster capacity worker')
ax2=sb.distplot(worker0, hist=False, rug=False, color='g', label='0.05x cluster capacity worker')
ax3=sb.distplot(worker1, hist=False, rug=False, color='b', label='0.69x cluster capacity worker')
xticks = [v for v in range(0,8,1)]
plt.legend(loc='upper right', fontsize=18)
plt.xticks(xticks)
plt.xlabel('mean training time per-iteration (seconds)', fontsize=18)
plt.ylabel('probability density of training time per-iteration', fontsize=18)
# plt.title('gradient adjustment with variable batching', fontsize=18)
plt.title('vanilla TF with constant batching', fontsize=18)
plt.show()