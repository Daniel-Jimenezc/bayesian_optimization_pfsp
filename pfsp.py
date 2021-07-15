import csv
import argparse
import time
import warnings
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from kernels import PermutationKernel

from scipy.optimize import minimize


def acq_max(optimizer, ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    x_tries = np.argsort(x_tries)
    # x_tries = np.array([elem for elem in x_tries if elem not in optimizer.space])
    ys = ac(x_tries, gp=gp, y_max=y_max)

    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])



class CustomOptimizer(BayesianOptimization):
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
        suggestion = acq_max(self,
                             ac=utility_function.utility,
                             gp=self._gp,
                             y_max=self._space.target.max(),
                             bounds=self._space.bounds,
                             random_state=self._random_state,
                             n_warmup=20000,
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
        c[0, 0] = self.p[int(sigma[0]), 0]
        for i in range(self.n_jobs):
            c[i, 0] = self.p[int(sigma[i]), 0] + c[i - 1, 0]
        for j in range(self.m_machines):
            c[0, j] = self.p[int(sigma[0]), j] + c[0, j - 1]
        for i in range(1, self.n_jobs):
            for j in range(1, self.m_machines):
                c[i, j] = self.p[int(sigma[i]), j] + max(c[i - 1, j], c[i, j - 1])
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
        permutation = list(permutation.values())
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
                if job == 200 and machine == 5:
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

    def execute_enhanced(self, file, seed, it=1000, rep=0, alpha=2.5, kappa=2.5, xi=0.0):
        self.n_jobs, self.m_machines, self.p = self.read_fsp_file(file)
        with open("results/instances/" + file[19:-4] + "_" + str(seed) + "_" + str(alpha) + "_" + str(kappa) + ".csv",
                  "w") as fr:
            writer = csv.writer(fr, delimiter=",", lineterminator="\n")
            writer.writerow(['target', 'best_result'])

        target_results = []
        points_results = []

        # Bounds of each variable
        pbounds = self.generate_bounds(self.n_jobs, -2*self.n_jobs, 2*self.n_jobs)

        # Bayesian Optimizer
        optimizer = CustomOptimizer(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=seed,
        )

        # Kernel
        # optimizer.set_gp_params(kernel=PermutationKernel())
        optimizer.set_gp_params(kernel=RBF(alpha))

        # Adquisition function
        utility = UtilityFunction(kind="ucb", kappa=kappa, xi=xi)
        results = []
        best_results = []
        best_result = -999999999999
        # Bayesian Optimization with Gaussian Process

        next_point = optimizer.suggest(utility)
        next_point = self.generate_next_point(self.random_key_enhanced(**next_point))
        target = self.black_box_function_enhanced(next_point)
        optimizer.register(params=next_point, target=target)

        for _ in range(it):
            t_ini = time.time()
            # optimizer.set_gp_params(kernel=PermutationKernel(0.001))
            next_point = optimizer.suggest(utility)
            print(next_point)
            # optimizer.set_gp_params(kernel=PermutationKernel(1000))
            # next_point = self.random_key_enhanced(**next_point_R)
            t_end = time.time() - t_ini
            # print(t_end)
            target = self.black_box_function_enhanced(next_point)
            optimizer.register(params=next_point, target=target)
            points_results.append(next_point)
            target_results.append(int(target))
            if optimizer.max['target'] > best_result:
                best_result = optimizer.max['target']
            with open(
                    "results/instances/" + file[19:-4] + "_" + str(seed) + "_" + str(alpha) + "_" + str(kappa) + ".csv",
                    "a") as fr:
                writer = csv.writer(fr, delimiter=",", lineterminator="\n")
                writer.writerow([int(target), int(best_result)])
            print(target, best_result)
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
    parser.add_argument("benchmark")
    parser.add_argument("jobs")
    parser.add_argument("machines")
    parser.add_argument("seed")  # Instance
    parser.add_argument("alpha")
    parser.add_argument("kappa")
    parser.add_argument("iterations")
    args = parser.parse_args()
    pfsp = PFSP()
    if args.benchmark == 'taillard':
        file_path = 'taillard_benchmark/tai'
    else:
        file_path = ''

    file = "{}{}_{}_{}.fsp".format(file_path, args.jobs, args.machines, args.seed)
    seed = int(args.seed)
    alpha = float(args.alpha.replace(",", "."))
    kappa = float(args.kappa.replace(",", "."))
    iterations = int(args.iterations)
    # file = 'taillard_benchmark/tai20_5_0.fsp'
    # seed = 0
    # it = 1000
    # alpha = 1.0
    # kappa = 10.0
    # rep = 1

    # print(args.file, int(args.seed), float(args.nu), float(args.kappa))
    print(file, seed, alpha, kappa)
    best_result, seed, results, best_results = pfsp.execute_enhanced(file, seed=seed, it=iterations, alpha=42,
                                                                     kappa=2000)
    print('--------------------')
    best_result, seed, results, best_results = pfsp.execute_enhanced(file, seed=seed, it=iterations, alpha=1000.0,
                                                                     kappa=1000)
