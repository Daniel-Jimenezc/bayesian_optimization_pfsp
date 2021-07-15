import argparse
import os
import numpy as np
import pfsp
import csv
import time

if __name__ == '__main__':
    a = pfsp.PFSP()
    jobs = (50,)
    machines = (10,)
    instances = (0, 1, 2,3,4,5,6,7,8,9)
    reps = 1
    it = 200

    files = a.generate_list_of_files_taillard(jobs, machines, instances)

    p = pfsp.PFSP()
    with open("results/taillard_results.csv", "w") as fp:
        writer = csv.writer(fp, delimiter=",", lineterminator="\n")
        # writer.writerow(["your", "header", "foo"])  # write header
        writer.writerow(['file', 'best_result','alpha','kappa','seed'])

    alphas = np.arange(0.1,3.1,0.1)
    kappas = [10e-5,10e-4,10e-3,10e-2,1,10,100,1000,10000]

    k = 0
    for file in files:
        for i in range(reps):
            print(file)
            for alpha in alphas:
                for kappa in kappas:
                    print(alpha,kappa)
                    with open("results/instances/"+file[19:-4]+"_"+str(i)+".csv", "w") as fr:
                        writer = csv.writer(fr, delimiter=",", lineterminator="\n")
                        writer.writerow(['target', 'best_result'])
                    ini = time.time()
                    best_result, seed, results, best_results = p.execute_enhanced(file, seed=k, it=it, alpha=alpha, kappa=kappa,rep=i)
                    end = time.time() - ini
                    with open("results/taillard_results.csv", "a") as fp:
                        writer = csv.writer(fp, delimiter=",", lineterminator="\n")
                        # writer.writerow(["your", "header", "foo"])  # write header
                        writer.writerow([file, best_result, alpha, kappa, k])
                    k += 1
