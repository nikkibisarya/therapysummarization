
import numpy as np
import matplotlib.pyplot as plt

loss = [1.0761, 0.8476, 0.7516, 0.6956, 0.6562, 0.6243, 0.5985, 0.5765, 0.5586, 0.5427, 0.5315, 0.5169, 0.5089, 0.4994,
        0.4923, 0.4866, 0.4806, 0.4763, 0.4708, 0.4707]

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.plot(np.arange(len(loss)), loss)
plt.savefig('MISCloss.png')

