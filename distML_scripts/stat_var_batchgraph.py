import matplotlib.pyplot as plt
import numpy as np

barwidth=0.05
yticks = [v for v in range(0, 1500, 100)]
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