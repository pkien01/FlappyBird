import matplotlib.pyplot as plt
import math
import numpy as np

epochs = []
losses = []
scores = []
distances = []

with open("q_log.txt", "r") as logfile:
    for line in logfile:
        if 'Q loss' in line and 'score' in line and 'distance survived' in line:
            epochs.append(int(line.split(' ')[1].split('/')[0]))  
            losses.append(float(line.split(':')[1].split(',')[0].strip()))
            scores.append(math.log(int(line.split(':')[2].split(',')[0].strip()) + 1))
            distances.append(math.log(int(line.split(':')[3].strip()) + 1))

plt.plot(epochs, losses, c = (np.random.random(), np.random.random(), np.random.random()))
plt.xlabel('Epoch #')
plt.ylabel('loss')
plt.savefig("epoch_vs_loss.png")
plt.clf()

plt.plot(epochs, scores, c = (np.random.random(), np.random.random(), np.random.random()))
plt.xlabel('Epoch #')
plt.ylabel('ln(score)')
plt.savefig("epoch_vs_score.png")
plt.clf()

plt.plot(epochs, distances, c = (np.random.random(), np.random.random(), np.random.random()))
plt.xlabel('Epoch #')
plt.ylabel('ln(frames_survived)')
plt.savefig("epoch_vs_dist.png")