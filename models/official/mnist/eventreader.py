import tensorflow as tf
import csv

metric_dict = {}
step_dict = {}
metric_index = 0
step_index = 0

#eventfiles = '/home/styagi/expdistresnet/23/'
eventfiles = '/home/styagi/prateeks/eval/'
for e in tf.train.summary_iterator(eventfiles + 'events.out.tfevents.1569365847.b2d5346dfe64'):
#for e in tf.train.summary_iterator(eventfiles):
	for v in e.summary.value:
		if v.tag == 'accuracy':
			print('accuracy is ', v.simple_value)
			metric_dict[metric_index] = v.simple_value
			metric_index+=1

		if v.tag == 'loss':
			print('current step is ', v.simple_value)
			step_dict[step_index] = v.simple_value
			step_index+=1

with open(eventfiles + 'summaryscalers.csv', mode='w') as metric_file:
	wrtr = csv.writer(metric_file, delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for k in metric_dict:
		wrtr.writerow([step_dict[k], metric_dict[k]])
