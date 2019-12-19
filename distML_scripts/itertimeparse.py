import os
experiment_config = '/Users/sahiltyagi/Desktop/periteration/asporggradvarbatch/tf-master-0'
for f in os.listdir(experiment_config):
	f = os.path.join(experiment_config, f)
	file = open(f,'r')
	out_file = "master0.csv"
	f2 =  os.path.join(experiment_config, out_file)
	out = open(f2, 'a')
	for line in file:
		if "@sahiltyagi iteration time on given worker" in line:
			out.write(line.split()[7] + '\n')
	out.close()
	file.close()