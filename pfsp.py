import csv
import argparse
import warnings
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.util import acq_max
from bayes_opt import UtilityFunction
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ExpSineSquared, RBF

from kernels import PermutationKernel

class CustomOptimization(BayesianOptimization):
    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state,
            n_warmup=10000,
            n_iter=0
        )

        return self._space.array_to_params(suggestion)

class PFSP:
    def __init__(self):
        self.p = None
        self.n_jobs = None
        self.m_machines = None

    def C_init(self, sigma):
        c = np.zeros(self.p.shape)
        c[0, 0] = self.p[sigma[0], 0]
        for i in range(self.n_jobs):
            c[i, 0] = self.p[sigma[i], 0] + c[i - 1, 0]
        for j in range(self.m_machines):
            c[0, j] = self.p[sigma[0], j] + c[0, j - 1]
        for i in range(1, self.n_jobs):
            for j in range(1, self.m_machines):
                c[i, j] = self.p[sigma[i], j] + max(c[i - 1, j], c[i, j - 1])
        return c

    def F(self, permutation):
        summation = 0
        c = self.C_init(permutation)
        for i in range(len(permutation)):
            summation -= c[i, self.m_machines - 1]
        return summation

    def random_key(self, v):
        permutation = np.argsort(v)
        return permutation

    def random_key_enhanced(self, **kwargs):
        data = np.fromiter(kwargs.values(), dtype=float)
        permutation = self.random_key(data)
        return permutation

    def black_box_function(self, **kwargs):
        data = np.fromiter(kwargs.values(), dtype=float)
        permutation = self.random_key(data)
        cost = self.F(permutation)
        return cost

    def black_box_function_enhanced(self, permutation):
        cost = self.F(permutation)
        return cost

    def read_fsp_file(self, filename):
        with open(filename) as f:
            content = f.readlines()

        # Extract number of jobs and machines from the second line
        header = list(map(int, ' '.join(content[1].split()).split()))
        n_jobs, m_machines = header[0], header[1]

        # Extract the processing times of jobs
        p_times_str = content[3:3 + int(m_machines)]
        p_matrix = []
        for row in p_times_str:
            p_matrix.append(list(map(int, ' '.join(row.split()).split())))
        p_times = np.matrix(p_matrix).transpose()

        return n_jobs, m_machines, p_times

    def generate_list_of_files_taillard(self, jobs=(20, 50, 100, 200, 500), machines=(5, 10, 20),
                                        instances=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
        list_of_files = []
        for job in jobs:
            for machine in machines:
                if job==200 and machine == 5:
                    continue
                if job == 500 and (machine == 5 or machine == 10):
                    continue
                for instance in instances:
                    file = "taillard_benchmark/tai{}_{}_{}.fsp".format(job, machine, instance)
                    list_of_files.append(file)
        return list_of_files

    def generate_list_of_files_random(self, jobs=(250, 300, 350, 400, 450), machines=(10, 20),
                                      instances=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
        list_of_files = []
        for job in jobs:
            for machine in machines:
                for instance in instances:
                    file = "random_benchmark/cebe{}_{}_{}.fsp".format(job, machine, instance)
                    list_of_files.append(file)
        return list_of_files

    def generate_bounds(self, n, lower_bound=0, upper_bound=1):
        i = 0
        pbounds = {}
        while i < n:
            xi = 'x' + str(i)
            pbounds[xi] = (lower_bound, upper_bound)
            i += 1
        return pbounds

    def generate_next_point(self, data):
        i = 0
        next_point = {}
        while i < len(data):
            xi = 'x' + str(i)
            next_point[xi] = data[i]
            i += 1
        return next_point

    def execute_kalimero(self, file, seed, it=1000, alpha=2.5, kappa=2.5, xi=0.0):
        self.n_jobs, self.m_machines, self.p = self.read_fsp_file(file)
        target_results = []
        points_results = []

        # Bounds of each variable
        pbounds = self.generate_bounds(self.n_jobs)

        # Bayesian Optimizer
        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=seed,
        )

        # Kernel
        optimizer.set_gp_params(kernel=PermutationKernel(alpha=alpha))

        # Adquisition function
        utility = UtilityFunction(kind="ucb", kappa=2.5, xi=xi)

        # Bayesian Optimization with Gaussian Process
        for _ in range(it):
            next_point = optimizer.suggest(utility)
            target = self.black_box_function(**next_point)
            optimizer.register(params=next_point, target=target)
            points_results.append(next_point)
            target_results.append(int(target))

        savefile = 'results/' + file[21:-4]+'__{}_{}_{}'.format(alpha, kappa, seed)
        row = (file, int(optimizer.max['target']), seed, alpha, kappa)
        np.save(savefile, row)


    def random_execution(self, file, size, it):
        max_cost = -999999999
        self.n_jobs, self.m_machines, self.p = self.read_fsp_file(file)
        for _ in range(it):
            vector = np.random.rand(size)
            permutation = self.random_key(vector)
            cost = self.F(permutation)
            if cost>max_cost:
                max_cost = cost
        return max_cost

    def execute(self, file, seed, it=1000, rep=0, alpha=2.5, kappa=2.5, xi=0.0):
        self.n_jobs, self.m_machines, self.p = self.read_fsp_file(file)

        target_results = []
        points_results = []

        # Bounds of each variable
        pbounds = self.generate_bounds(self.n_jobs)

        # Bayesian Optimizer
        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=seed,
        )

        # Kernel
        # optimizer.set_gp_params(kernel=PermutationKernel())
        optimizer.set_gp_params(kernel=PermutationKernel())

        # Adquisition function
        utility = UtilityFunction(kind="ucb", kappa=kappa, xi=xi)
        results = []
        best_results=[]
        best_result = -999999999999
        # Bayesian Optimization with Gaussian Process
        for _ in range(it):
            next_point = optimizer.suggest(utility)
            target = self.black_box_function(**next_point)
            optimizer.register(params=next_point, target=target)
            points_results.append(next_point)
            target_results.append(int(target))
            if optimizer.max['target']>best_result:
                best_result = optimizer.max['target']
            with open("results/instances/"+file[19:-4]+"_"+str(rep)+".csv", "a") as fr:
                writer = csv.writer(fr, delimiter=",", lineterminator="\n")
                writer.writerow([int(target), int(best_result)])
        return optimizer.max['target'], seed, results, best_results


    def execute_enhanced(self, file, seed, it=1000, rep=0, alpha=2.5, kappa=2.5, xi=0.0):
        self.n_jobs, self.m_machines, self.p = self.read_fsp_file(file)

        target_results = []
        points_results = []

        # Bounds of each variable
        pbounds = self.generate_bounds(self.n_jobs)

        # Bayesian Optimizer
        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=seed,
        )

        # Kernel
        # optimizer.set_gp_params(kernel=PermutationKernel())
        optimizer.set_gp_params(kernel=PermutationKernel())

        # Adquisition function
        utility = UtilityFunction(kind="ucb", kappa=kappa, xi=xi)
        results = []
        best_results=[]
        best_result = -999999999999
        # Bayesian Optimization with Gaussian Process
        for _ in range(it):
            next_point = optimizer.suggest(utility)
            next_point = self.random_key_enhanced(**next_point)
            print(next_point)
            target = self.black_box_function_enhanced(next_point)
            optimizer.register(params=self.generate_next_point(next_point), target=target)
            points_results.append(next_point)
            target_results.append(int(target))
            if optimizer.max['target']>best_result:
                best_result = optimizer.max['target']
            with open("results/instances/"+file[19:-4]+"_"+str(rep)+".csv", "a") as fr:
                writer = csv.writer(fr, delimiter=",", lineterminator="\n")
                writer.writerow([int(target), int(best_result)])
        return optimizer.max['target'], seed, results, best_results

    def execute_taillard_benchmarks(self, jobs=(20, 50, 100, 200, 500), machines=(5, 10, 20),
                                    instances=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), seed=0, it=1000):
        with open("taillard_results.csv", "w") as fp:
            writer = csv.writer(fp, delimiter=",", lineterminator="\n")
            # writer.writerow(["your", "header", "foo"])  # write header
            writer.writerow(['file', 'points_results', 'target_results', 'best_result', 'seed', 'n_it'])
        list_of_files = self.generate_list_of_files_taillard(jobs, machines, instances)
        for file in list_of_files:
            file, points_results, target_results, best_result, seed, it = self.execute(file, seed, it)
            # Write CSV file
            with open("taillard_results.csv", "a") as fp:
                writer = csv.writer(fp, delimiter=",", lineterminator="\n")
                writer.writerow((file, int(best_result['target']), seed, it))

    def execute_random_benchmarks(self, jobs=(250, 300, 350, 400, 450), machines=(10, 20),
                                  instances=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), seed=0, it=1000):
        with open("random_results.csv", "w") as fp:
            writer = csv.writer(fp, delimiter=",", lineterminator="\n")
            # writer.writerow(["your", "header", "foo"])  # write header
            writer.writerow(['file', 'target_results', 'best_result', 'seed', 'n_it'])
        list_of_files = self.generate_list_of_files_random(jobs, machines, instances)
        for file in list_of_files:
            file, points_results, target_results, best_result, seed, it = self.execute(file, seed, it)
            # Write CSV file
            with open("random_results.csv", "a") as fp:
                writer = csv.writer(fp, delimiter=",", lineterminator="\n")
                writer.writerow((file, int(best_result['target']), seed, it))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("seed")
    parser.add_argument("nu")
    parser.add_argument("kappa")
    args = parser.parse_args()
    pfsp = PFSP()
    file = 'taillard_benchmark/tai20_5_0.fsp'
    seed = 0
    it = 200
    alpha = 1.0
    kappa = 10.0
    rep = 1

    #print(args.file, int(args.seed), float(args.nu), float(args.kappa))
    best_result, seed, results, best_results = pfsp.execute_enhanced(file, seed=seed, it=it, alpha=alpha, kappa=kappa,rep=rep)
