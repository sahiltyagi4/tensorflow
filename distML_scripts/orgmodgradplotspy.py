import matplotlib.pyplot as plt
import numpy as np

h_05_mod = [90.41,90.62,91.66]
h_05_modtime = [255,158,1368]
h_05_modbatchsize = [128,64,128]
h_05_modsync = ['ASP','ASP', 'BSP']

h_05_org = [89.57, 90.61,91.73]
h_05_orgtime = [268,162,1312]
h_05_orgbatchsize = [128,64,128]
h_05_orgsync = ['ASP','ASP','BSP']


h_925_mod = [79.79,92.96]
h_925_modtime = [377,714]
h_925_modbatchsize = [128,64]
h_925_modsync = ['ASP','BSP']

h_925_org = [87.02,90.6]
h_925_orgtime = [397,5690]
#2nd PART batch-size is actually 128, get results for this soon!
h_925_orgbatchsize = [128,64]
h_925_orgsync = ['ASP','BSP']

h_05_orgvarbatch = [90.28,89.93,0]
h_05_orgvarbatchtime = [252,156,0]
h_05_orgvarbatchsize = [128,64,0]
h_05_orgvarbatchsync = ['ASP','ASP','']

h_925_orgvarbatch = [74.98,0]
h_925_orgvarbatchtime = [370,0]
h_925_orgvarbatchsize = [128,0]
h_925_orgvarbatchsync = ['ASP','']

# plt.xlabel('computation time (minutes)', fontsize=25)
# plt.ylabel('Test accuracy %', fontsize=25)
# plt.plot(h_05_modtime, h_05_mod, color='b', linewidth=4, marker='o', label='gradient adjustment')
# plt.plot(h_05_orgtime, h_05_org, color='r', linewidth=4, marker='x', label='vanilla TF', mew=4, ms=8)
# #plt.plot(h_925_modtime, h_925_mod, color='b', linewidth=2, marker='o', label='gradient adjustment')
# #plt.plot(h_925_orgtime, h_925_org, color='r', linewidth=2, marker='x', label='vanilla TF',mew=4, ms=8)
# plt.legend(loc='lower right', fontsize=25)
# plt.title('h-level = 0.5', loc='center', fontsize=20)
# plt.show()

barwidth=3
# r1 = [119, 60, 129]
r1 = [59,125]
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]

#rec1 = plt.bar(r1, h_05_mod, color='b', label='gradient adjustment', width=barwidth)
#rec2 = plt.bar(r2, h_05_org, color='r', label='vanilla TF with constant batch-size', width=barwidth)
#rec3 = plt.bar(r3, h_05_orgvarbatch, color='g', label='vanilla TF with variable batch-size', width=barwidth)

rec1 = plt.bar(r1, h_925_mod, color='b', label='gradient adjustment', width=barwidth)
rec2 = plt.bar(r2, h_925_org, color='r', label='vanilla TF with constant batch-size', width=barwidth)
rec3 = plt.bar(r3, h_925_orgvarbatch, color='g', label='vanilla TF with variable batch-size', width=barwidth)

i=0
for rec in rec1:
	height = rec.get_height()
	print(height)
	#for accuracy
	xy = (rec.get_x() + rec.get_width() / 2, height)
	xytext = (rec.get_x() + rec.get_width() / 2, height + 4)
	#for time
	#xytext = (rec.get_x() + rec.get_width() / 2, height + 180)
	plt.annotate(h_925_modsync[i], xy=xy, xytext=xytext, rotation=90)
	i+=1

i=0
for rec in rec2:
	height = rec.get_height()
	print(height)
	xy = (rec.get_x() + rec.get_width() / 2, height)
	xytext = (rec.get_x() + rec.get_width() / 2, height + 4)
	#for time
	#xytext = (rec.get_x() + rec.get_width() / 2, height + 180)
	plt.annotate(h_925_orgsync[i], xy=xy, xytext=xytext, rotation=90)
	i+=1

i=0
for rec in rec3:
	height = rec.get_height()
	print(height)
	xy = (rec.get_x() + rec.get_width() / 2, height)
	xytext = (rec.get_x() + rec.get_width() / 2, height + 4)
	#for time
	#xytext = (rec.get_x() + rec.get_width() / 2, height + 180)
	plt.annotate(h_925_orgvarbatchsync[i], xy=xy, xytext=xytext, rotation=90)
	i+=1

plt.xlabel('average cluster batch-size', fontsize=25)
plt.ylabel('Test accuracy %', fontsize=25)
#plt.ylabel('computation time (minutes)', fontsize=25)
plt.legend(loc='upper left', fontsize=15)
#yticks = [v for v in range(0,6400, 400)]
#plt.yticks(yticks)

xticks=[0,32,64,128]
plt.xticks(xticks)
plt.title('h-level = 0.925', loc='center', fontsize=20)
plt.show()