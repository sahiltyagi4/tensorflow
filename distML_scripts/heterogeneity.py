import matplotlib.pyplot as plt

h_levels = [0,0.5,0.926]
accuracy_asp = [89.69, 89.03, 89.02]
computation_time_asp = [275, 276, 308]
accuracy_bsp = [90.74, 91.73, 90.6]
computation_time_bsp = [915, 1312, 5690]

# plt.plot(h_levels, accuracy_asp, color='b', label='ASP', marker='o')
# plt.plot(h_levels, accuracy_bsp, color='r', label='BSP', marker='D')
plt.plot(h_levels, computation_time_asp, color='b', label='ASP', marker='o')
plt.plot(h_levels, computation_time_bsp, color='r', label='BSP', marker='D')
plt.legend(loc='upper left', fontsize=25)
plt.xlabel('h-level', fontsize=25)
plt.ylabel('computation time (min.)', fontsize=25)
plt.title('computation time in ASP and BSP at different h-levels', fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()