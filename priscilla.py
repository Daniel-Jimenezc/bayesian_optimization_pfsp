import os
import numpy as np

jobs = [20,50,100]
machines = [5,10,20]
instances = [0,1,2,3,4,5,6,7,8,9]
seeds = [0,1,2,3,4,5,6,7,8,9]
alphas = np.arange(0.1,3.1,0.1)
kappas = [1e-5,0.0001,0.001,0.1,1.0,10.0,100.0,1000.0,10000.0]

results_50_10 = {}
def best_configuration(jobs=50,machines=10):
    for file in os.listdir('Priscilla/instances'):
        j,m,i,s,a,k = file[3:-4].split('_')
        if j==str(jobs) and m == str(machines):
            with open('Priscilla/instances/'+file, 'rb') as f:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
                last_line = f.readline().decode()
            target = (last_line.strip().split(',')[1])
            config = a+'_'+k
            if config in results_50_10:
                results_50_10[config] += int(target)
            else:
                results_50_10[config] = int(target)

    print(results_50_10)
if __name__ == '__main__':
    best_configuration()



