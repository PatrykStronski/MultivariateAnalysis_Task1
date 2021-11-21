import pandas as pd
import scipy

# KStest - if pvalue is small then it means that the model does not fit
# Cramer-von Mises test - 
def calculate_tests(kde_values: pd.Series, dist_name: str, params: tuple) -> dict:
    ks = scipy.stats.kstest(kde_values, dist_name, params)
    omega2 = scipy.stats.cramervonmises(kde_values, dist_name, params)
    print(f'FOR {dist_name}: Kolmogorov-Smirnoff test result {ks}, whereas Omega squared test (Cramér–von Mises) test {omega2}')
    return { 'dist': dist_name, 'ks': ks, 'omega': omega2}

def calculate_tests_v2(kde_values: pd.Series, estimated: pd.Series, dist_name: str, params: tuple) -> dict:
    ks = scipy.stats.kstest(kde_values, estimated)
    omega2 = scipy.stats.cramervonmises(kde_values, dist_name)
    print(f'FOR {dist_name}: Kolmogorov-Smirnoff test result {ks}, whereas Omega squared test (Cramér–von Mises) test {omega2}')
    return { 'dist': dist_name, 'ks': ks, 'omega': omega2}

def choose_best_fitness(test_results: list) -> tuple:
    best_fit_ks = { 'dist': None }
    best_fit_omega = { 'dist': None }
    for res in test_results:
        if res['ks'].pvalue == 0.0:
            1+1
        elif best_fit_ks['dist'] == None:
            best_fit_ks = res
        elif best_fit_ks['ks'].pvalue < res['ks'].pvalue:
            best_fit_ks = res
        
        if res['omega'].pvalue == 0.0:
            1+1
        elif best_fit_omega['dist'] == None:
            best_fit_omega = res
        elif best_fit_omega['omega'].pvalue < res['omega'].pvalue:
            best_fit_omega = res
    return (best_fit_ks, best_fit_omega)