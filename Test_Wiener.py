import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

n_individuals = 5

global noise
noise = np.zeros((n_individuals, 1))
print(noise)
global total_noise
total_noise = np.zeros((n_individuals, 1))  

def wiener_process_step(dt):
    dW = np.sqrt(dt) * np.random.randn(n_individuals, 1)
    #print(dW)
    global noise
    noise += dW
    global total_noise
    total_noise = np.hstack((total_noise, noise))



t = np.linspace(0, 1, 100)


for i in range(len(t)):
    wiener_process_step(1 / 100)
    print("Current noise:\n", noise)
    print("Total noise history:\n", total_noise)

print(total_noise.shape)

plt.plot(t, total_noise[0,:100], label='Individual 1')
plt.plot(t, total_noise[1,:100], label='Individual 2')
plt.plot(t, total_noise[2,:100], label='Individual 3')
plt.plot(t, total_noise[3,:100], label='Individual 4')
plt.plot(t, total_noise[4,:100], label='Individual 5')
plt.show()
