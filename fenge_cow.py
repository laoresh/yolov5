import os
import numpy as np
import matplotlib.pyplot as plt
path = r'E:\Shinelon\pycharm2020\pytorch_works\yolov5-5.0\runs\detect\exp\labels'
pathdir = os.listdir(path)
del pathdir[445]
L = len(pathdir)
y = []
for i in range(L):
    pathname = path + '/' + pathdir[i]
    yy = np.loadtxt(pathname)
    if len(yy) != 5:
        continue
    y.append(yy[2])
print(y)
plt.figure(1,(8,6),dpi=80)
x = [i for i in range(1745)]
plt.plot(x, y)
plt.show()
