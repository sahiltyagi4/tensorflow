## this program computes the average per-iteration time of the cluster comprising each node type in TF_CONFIG.
##Additionally, based on every node's average batch-size, we compute the time_scale of each node.
import os
# dir='/home/styagi/nfs2data/'
dir='/home/styagi/1/'
batchsizelist = 'batchsizelist.txt'
avgtime = []
files=['tf-master-0/steplogs.txt', 'tf-worker-0/steplogs.txt', 'tf-worker-1/steplogs.txt']
for f in files:
	sum = 0
	count = 0
	f = os.path.join(dir, f)
	file = open(f, 'r')
	for line in file:
		if "@sahiltyagi iteration time on given worker" in line:
			sum = sum + line.split()[7]
			count = count + 1

	file.close()
	node_avgtime = (sum/count)
	avgtime.append(node_avgtime)
	print('for file ' + str(f) + ' the average per-iteration time is ' + str(node_avgtime))

cluster_totaltime = 0
total_averagetime = 0
for i in avgtime:
	cluster_totaltime = cluster_totaltime + i

total_averagetime = (cluster_totaltime/len(avgtime))

time_scale = []
for t in avgtime:
	time_scale.append(t/total_averagetime)

f = os.path.join(dir, batchsizelist)
file = open(f, 'r')
batchlist = []
for line in file:
	print('old batch_list: ' + line)
	for batchsize in line.split(','):
		batchlist.append(batchsize.replace('[', '').replace(']', ''))

file.close()

new_batch_list = '['
for i in range(0, len(time_scale)):
	new_batch_list = new_batch_list + str(time_scale[i] * batchlist[i]) + ','

new_batch_list = new_batch_list[0 : len(new_batch_list) -1] + ']'
print('new batch_list is: ' + str(new_batch_list))

file = open(f, 'w')
file.write(new_batch_list + '\n')
file.close()