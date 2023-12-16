import matplotlib.pyplot as plt
import math
import numpy as np

generations = []
max_scores = []
max_distances = []

with open("g_log.txt", "r") as logfile:
    for line in logfile:
        if 'max score' in line and 'max distance survived' in line:
            generations.append(int(line.split(' ')[1][:-2]))  
            max_scores.append(math.log(int(line.split('=')[1].split(',')[0].strip()) + 1))
            max_distances.append(math.log(int(line.split('=')[2].strip()) + 1))


plt.plot(generations, max_scores, c = (np.random.random(), np.random.random(), np.random.random()))
plt.xlabel('Generation #')
plt.ylabel('ln(max_score)')
plt.savefig("gen_vs_score.png")
plt.clf()

plt.plot(generations, max_distances, c = (np.random.random(), np.random.random(), np.random.random()))
plt.xlabel('Generation #')
plt.ylabel('ln(max_frames_survived)')
plt.savefig("gen_vs_dist.png")