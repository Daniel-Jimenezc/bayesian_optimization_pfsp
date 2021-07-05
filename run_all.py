import argparse
import os
import numpy as np
import pfsp
import csv
import time

if __name__ == '__main__':
    a = pfsp.PFSP()
    jobs = (50, 100, 200, 500)
    machines = (5, 10, 20)
    instances = (0, 1, 2)
    reps = 1
    it = 500

    files = a.generate_list_of_files_taillard(jobs, machines, instances)

    p = pfsp.PFSP()
    with open("results/taillard_results.csv", "w") as fp:
        writer = csv.writer(fp, delimiter=",", lineterminator="\n")
        # writer.writerow(["your", "header", "foo"])  # write header
        writer.writerow(['file', 'best_result', 'rep','seed'])

    alpha = 1.0
    kappa = 10.0

    k = 0
    for file in files:
        for i in range(reps):
            print(file)
            with open("results/instances/"+file[19:-4]+"_"+str(i)+".csv", "w") as fr:
                writer = csv.writer(fr, delimiter=",", lineterminator="\n")
                writer.writerow(['target', 'best_result'])
            ini = time.time()
            best_result, seed, results, best_results = p.execute_enhanced(file, seed=k, it=it, alpha=alpha, kappa=kappa,rep=i)
            end = time.time() - ini
            with open("results/taillard_results.csv", "a") as fp:
                writer = csv.writer(fp, delimiter=",", lineterminator="\n")
                # writer.writerow(["your", "header", "foo"])  # write header
                writer.writerow([file, best_result,i, k])
            k += 1
