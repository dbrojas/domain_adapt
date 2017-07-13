import numpy as np
import pandas as pd
from math import factorial
import matplotlib.pyplot as plt
from pylab import *

# savitzky-golay filter for loss/acc smoothing
def sg_filter(y, window_size=101, order=3, deriv=0, rate=1):
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


# load training data (clf loss, dis loss, and accuracy)
df1 = pd.read_csv('/home/fernando_bartolome3/bilt_acc0.169_default.csv')
df4 = pd.read_csv('/home/fernando_bartolome3/bilt_acc0.142_selective50.csv')
df3 = pd.read_csv('/home/fernando_bartolome3/bilt_acc0.157_iter100.csv')


# plot accuracy
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

ax.plot(range(len(df1)), sg_filter(np.array(df1.acc)), 'r',label='default')
ax.plot(range(len(df4)), sg_filter(np.array(df4.acc)), 'b',label='selective')
ax.plot(range(len(df3)), sg_filter(np.array(df3.acc)), 'g',label='iterative')

ax.plot(range(len(df3)), np.array(df3.acc), 'g', linewidth=0.8, alpha=0.2)
ax.plot(range(len(df1)), np.array(df1.acc), 'r', linewidth=0.8, alpha=0.2)
ax.plot(range(len(df4)), np.array(df4.acc), 'b', linewidth=0.8, alpha=0.2)

ax.plot(range(len(df1)), np.repeat(0.184, len(df1)), 'k--',label='source-only', linewidth=0.8)
ax.plot(range(len(df1)), np.repeat(0.663, len(df1)), 'k-',label='target-only', linewidth=0.8)

plt.title('Cabauw to De Bilt')
plt.xlabel('Batches seen')
plt.ylabel('Accuracy')


# plot classifier loss 
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

ax.plot(range(len(df1)), sg_filter(np.array(df1.c_loss)), 'r', label='default')
ax.plot(range(len(df4)), sg_filter(np.array(df4.c_loss)), 'b', label='selective')
ax.plot(range(len(df3)), sg_filter(np.array(df3.c_loss)), 'g',label='iterative')

ax.plot(range(len(df3)), np.array(df3.c_loss), 'g', linewidth=0.8, alpha=0.2)
ax.plot(range(len(df1)), np.array(df1.c_loss), 'r', linewidth=0.8, alpha=.2)
ax.plot(range(len(df4)), np.array(df4.c_loss), 'b', linewidth=0.8, alpha=.2)

plt.title('Cabauw to De Bilt')
plt.xlabel('Batches seen')
plt.ylabel('Mean Squared Error')


# plot discriminator loss
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

ax.plot(range(len(df1)), sg_filter(np.array(df1.d_loss)), 'r', label='default')
ax.plot(range(len(df4)), sg_filter(np.array(df4.d_loss)), 'b', label='selective')
ax.plot(range(len(df3)), sg_filter(np.array(df3.d_loss)), 'g',label='iterative')

ax.plot(range(len(df3)), np.array(df3.d_loss), 'g', linewidth=0.8, alpha=0.2)
ax.plot(range(len(df1)), np.array(df1.d_loss), 'r', linewidth=0.8, alpha=.2)
ax.plot(range(len(df4)), np.array(df4.d_loss), 'b', linewidth=0.8, alpha=.2)

plt.title('Cabauw to De Bilt')
plt.xlabel('Batches seen')
plt.ylabel('Binary cross-entropy')


