import pandas as pd
import scipy

# KStest - if pvalue is small then it means that the model does not fit
# Cramer-von Mises test - 
def calculate_tests(kde_values: pd.Series, dist_name: str, params: tuple):
    ks = scipy.stats.kstest(kde_values, dist_name, params)
    omega2 = scipy.stats.cramervonmises(kde_values, dist_name, params)
    print(f'FOR {dist_name}: Kolmogorov-Smirnoff test result {ks}, whereas Omega squared test (Cramér–von Mises) test {omega2}')