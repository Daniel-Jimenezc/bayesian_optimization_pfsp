import argparse
import os
import numpy as np
import pfsp
import csv
import time

if __name__ == '__main__':
    a = pfsp.PFSP()
    jobs = (20,)
    machines = (10,)
    instances = (0, 1, 2)
    reps = 5

    files = a.generate_list_of_files_taillard(jobs, machines, instances)

    #os.system('./lanzar_pfsp.sh {} {} {} {}'.format(files[0], 0, 1.0, 1.0))
    p = pfsp.PFSP()
    with open("results/taillard_results.csv", "w") as fp:
        writer = csv.writer(fp, delimiter=",", lineterminator="\n")
        # writer.writerow(["your", "header", "foo"])  # write header
        writer.writerow(['file', 'best_result', 'seed', 'alpha'])
    k = 0
    for file in files:
        for alpha in np.linspace(0.3, 3.3, 11):
            for i in range(reps):
                print(file, alpha, i)
                # python kalimero_pfsp.py file seed nu kappa
                # a.execute_kalimero(file=file, seed=i, it=10, nu=nu, kappa=kappa)
                # os.system('python3 pfsp.py {} {} {} {}'.format(file, i, nu, kappa))
                ini = time.time()
                best_result, seed = p.execute(file,seed=k,it=50,alpha=alpha,kappa=1.0)
                end = time.time() - ini
                print(end)
                with open("results/taillard_results.csv", "a") as fp:
                    writer = csv.writer(fp, delimiter=",", lineterminator="\n")
                    # writer.writerow(["your", "header", "foo"])  # write header
                    writer.writerow([file, best_result, k,alpha])
                k+=1

