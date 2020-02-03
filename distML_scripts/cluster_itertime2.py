import os

#when adding to estimator source code, go one directory above
dir = '/extra/1/'
timesteps = {}
avgtime = {}
for f in os.listdir(dir):
	f = os.path.join(dir, f)
	file = open(f, 'r')
	for line in file:
		if "@sahiltyagi iteration time on given worker is" in line:
			key = float(line.split()[17])
			if key in timesteps:
				list_of_time = timesteps[key]
			else :
				list_of_time = []

			timesteps[key] = list_of_time.append(float(line.split()[7]))

	file.close()
for key, val in timesteps.items():
	total = 0
	for t in val:
		total = total + t

	# average per-iteration time across all workers
	avgtime[key] = (total/len(val))