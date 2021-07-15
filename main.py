from pfsp import PFSP



pfsp = PFSP()
best_result, seed, results, best_results = pfsp.execute_enhanced(file, seed=seed, it=iterations, alpha=alpha, kappa=kappa)