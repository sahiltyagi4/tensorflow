import matplotlib.pyplot as plt
import numpy as np

##TTA accuracy threshold ASP: 85%
##TTA accuracy threshold BSP: 90%
barwidth=0.05
# yticks = [v for v in range(0, 6000, 200)]
# asp = [275, 276, 171, 308]
# bsp= [915, 1312, 797, 5690]

# with particular accuracy threshold for a single heterogenity level
# yticks = [v for v in range(0, 400, 40)]
# asp = [275, 270, 83, 80]
# bsp= [225, 257, 137, 351]
# heterogenity = [0, 0.5, 0.667, 0.926]
#accuracy thresholds are as follows for each heterogenity level
#accuracy_threshold=["89.69%", "89.00%", "88.35%", "81.38%"]
# r1 = heterogenity
# r2 = [x + barwidth for x in r1]
# for j in np.arange(0, 6000, 300):
# 	plt.axhline(j, color='grey', alpha=0.1)

# for j in np.arange(0, 1, 0.2):
# 	plt.axvline(j, color='grey', alpha=0.3)

# yticks = [v for v in range(0, 400, 50)]
# asp = [80,270,268,308]
# #bsp= [178, 255, 340, 1100]
# heterogenity = [0, 0.5, 0.667, 0.926]
# r1 = heterogenity
# plt.bar(r1, asp, color='b', label='ASP 89% accuracy', width=barwidth)
# #plt.bar(r1, bsp, color='b', label='BSP 88% accuracy', width=barwidth)
# plt.xlabel('Heterogenity', fontsize=25)
# plt.ylabel('Convergence time for 50K steps (minutes)', fontsize=25)
# plt.yticks(yticks, rotation='horizontal')
# plt.legend(loc='upper left', fontsize=25)
# plt.show()

yticks = [v for v in range(0, 1000, 100)]
static = [915,1312,797,308]
variable= [915,960,1188,305]
heterogenity = [0, 0.5, 0.667, 0.926]
r1 = heterogenity
r2 = [x + barwidth for x in r1]
plt.bar(r1, static, color='b', label='static batching', width=barwidth)
plt.bar(r2, variable, color='r', label='variable batching', width=barwidth)
plt.xlabel('Heterogenity', fontsize=25)
plt.ylabel('Convergence time for 50K steps (minutes)', fontsize=25)
plt.yticks(yticks, rotation='horizontal')
plt.legend(loc='upper left', fontsize=25)
plt.show()