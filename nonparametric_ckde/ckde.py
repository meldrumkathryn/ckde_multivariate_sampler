# Author: Kate Meldrum 
# class ckde: an initialized ckde class containing training data
# ckde.generate_sim_points(): allows user to create simulation points for conditional multivariate non-parametric distribtuion

import numpy as np
import pandas as pd
import sklearn.neighbors as sk
import sklearn.model_selection as model_selection 
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
import random 

import warnings 
warnings.filterwarnings("ignore")

class ckde: 

    def __init__(self, data, n=2, cv_bandwidth_params=[0.05, 0.5, 30, 3]):

        ''' Initialize Data for Conditional Kernel Density Estimation

        @param: data (dataframe): variable distributions to simulate, up to 10 variables
        @param: n (int): number of variables to simulate, sampler will select the first n columns of data. Defaults to 2. 
        @param: cv_bandwidth_params (list): [min grid search, max grid search, number of search terms, number of cv folds]

        '''

        self.cv_bandwidth_params = cv_bandwidth_params
        self.colnames = data.columns[0:n]
        self.n = n

        self.x1 = np.array(data.iloc[:, 0])
        self.x2 = np.array(data.iloc[:, 1])*100

        if n>2: 
            self.x3 = np.array(data.iloc[:, 2])*100
            self.x1_2 = np.vstack((self.x1, self.x2)).T

        if n>3: 
            self.x4 = np.array(data.iloc[:, 3])
            self.x1_3 = np.vstack((self.x1, self.x2, self.x3)).T

        if n>4:
            self.x5 = np.array(data.iloc[:, 4])
            self.x1_4 = np.vstack((self.x1, self.x2, self.x3, self.x4)).T

        if n>5:
            self.x6 = np.array(data.iloc[:, 5])
            self.x1_5 = np.vstack((self.x1, self.x2, self.x3, self.x4, self.x5)).T

        if n>6:
            self.x7 = np.array(data.iloc[:, 6])
            self.x1_6 = np.vstack((self.x1, self.x2, self.x3, self.x4, self.x5, self.x6)).T

        if n>7:
            self.x8 = np.array(data.iloc[:, 7])
            self.x1_7 = np.vstack((self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.x7)).T

        if n>8:
            self.x9 = np.array(data.iloc[:, 8])
            self.x1_8 = np.vstack((self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.x7, self.x8)).T

        if n>9:
            self.x10 = np.array(data.iloc[:, 9])
            self.x1_9 = np.vestack((self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.x7, self.x8, self.x9)).T

        if n>10: 

            print("ckde can only perform conditional sampling for up to 10 variables, any additional variables will be ignored")
            
            

    # helper funtions (user does not directly access)
    def cv_bandwidth(self):
        '''
        set an appropriate bandwidth parameter for the class instance
        '''
        
        # determine bandwidth via cross validation 

        #define bandwidth range to test (grid search params)
        test_bandwidths = np.linspace(self.cv_bandwidth_params[0], self.cv_bandwidth_params[1], self.cv_bandwidth_params[2])

        # use gridsearch to find best bandwidth
        test_grid = model_selection.GridSearchCV(sk.KernelDensity(), {'bandwidth': test_bandwidths}, cv = self.cv_bandwidth_params[3])
        test_grid.fit(np.vstack(self.x1))
        cv_bandwidth = test_grid.best_params_['bandwidth']
        self.bandwidth = cv_bandwidth

    def fit_x1(self):
        '''
        fit KDE for the first vairable using sklearn
        '''

        self.cv_bandwidth()
        # assign initial kde to class attribute
        self.x1_kde = sk.KernelDensity(kernel = 'gaussian', bandwidth = self.bandwidth).fit(np.vstack(self.x1))

    # use automatic sampling method for the sklearn estimate to sample from distribution 1 
    def sample_x1(self, num_samples): 
        '''
        sample from first KDE using existing sklearn method
        '''
        return self.x1_kde.sample(n_samples = num_samples)

    def sample_x_cond_X(self, x, fit, x_i, num_samples):
        '''
        inverse transform sampling function for sampling from KDEMultivaraiteConditional distributions for all following variables
        '''

        # plot cdf of x_i over 1000 values of x_i between min and max
        x_vals = np.linspace(min(x), max(x), 1000)

        # format 2D array where each point is x_i, y_i, for every z_val
        x_cond_vals = np.vstack(([np.full_like(x_vals, n) for n in x_i])).T
        cdf_vals = fit.cdf(x_vals, x_cond_vals)

        # pick a random cumulative probability value between 0 and 1 
        x_uni_samp = np.random.uniform(0, 1, num_samples)

        # return z value corresponding to that probability 
        return(np.interp(x_uni_samp, cdf_vals, x_vals))

    # these must be defined individually due to np. formatting error
    # fit each variable following the first one using KDEMultivaraiateConditional 
    def fit_x2_cond_x1(self):
        self.x2_cond_x1 = KDEMultivariateConditional(endog=self.x2, exog=self.x1, dep_type='c', indep_type='c', bw='cv_ml')

    def fit_x3_cond_x1_2(self):
        self.x3_cond_x1_2 = KDEMultivariateConditional(endog=self.x3, exog=self.x1_2, dep_type='c', indep_type='cc', bw='cv_ml')

    def fit_x4_cond_x1_3(self):
        self.x4_cond_x1_3 = KDEMultivariateConditional(endog=self.x4, exog=self.x1_3, dep_type='c', indep_type='ccc', bw='cv_ml')

    def fit_x5_cond_x1_4(self):
        self.x5_cond_x1_4 = KDEMultivariateConditional(endog=self.x5, exog=self.x1_4, dep_type='c', indep_type='cccc', bw='cv_ml')

    def fit_x6_cond_x1_5(self):
        self.x6_cond_x1_5 = KDEMultivariateConditional(endog=self.x6, exog=self.x1_5, dep_type='c', indep_type='ccccc', bw='cv_ml')

    def fit_x7_cond_x1_6(self):
        self.x7_cond_x1_6 = KDEMultivariateConditional(endog=self.x7, exog=self.x1_6, dep_type='c', indep_type='cccccc', bw='cv_ml')

    def fit_x8_cond_x1_7(self):
        self.x8_cond_x1_7 = KDEMultivariateConditional(endog=self.x8, exog=self.x1_7, dep_type='c', indep_type='ccccccc', bw='cv_ml')

    def fit_x9_cond_x1_8(self):
        self.x9_cond_x1_8 = KDEMultivariateConditional(endog=self.x9, exog=self.x1_8, dep_type='c', indep_type='cccccccc', bw='cv_ml')

    def fit_x10_cond_x1_9(self):
        self.x10_cond_x1_9 = KDEMultivariateConditional(endog=self.x10, exog=self.x1_9, dep_type='c', indep_type='ccccccccc', bw='cv_ml')

        

    # Main user fucntion: 
    def generate_sim_points(self, n_points):
        '''fit KDE distributions for all variables and produce conditionally sampled simualtion points

        @param: n_points(int): number of points ot generate
        '''

        #fit all kdes 
        self.fit_x1()
        self.fit_x2_cond_x1()

        if self.n>2: 
            self.fit_x3_cond_x1_2()

        if self.n>3:
            self.fit_x4_cond_x1_3()

        if self.n>4:
            self.fit_x5_cond_x1_4()

        if self.n>5:
            self.fit_x6_cond_x1_5()

        if self.n>6:
            self.fit_x7_cond_x1_6()

        if self.n>7:
            self.fit_x8_cond_x1_7()

        if self.n>8:
            self.fit_x9_cond_x1_8()

        if self.n>9:
            self.fit_x10_cond_x1_9()

        #initialize empty results dict
        results = {f'x{i}': [] for i in range(1, self.n+1)}

        print("Start Sampling")

        # generate points
        for i in range(n_points):

            # sample value for x1
            x1_i = self.sample_x1(1).item()
            results['x1'].append(x1_i)

            # sample value for x2 conditional on x1
            x2_i = self.sample_x_cond_X(self.x2, self.x2_cond_x1, [x1_i], 1).item()
            results['x2'].append(x2_i)

            if self.n>2:
                #sample value for x3 conditional on x1:x2
                x3_i = self.sample_x_cond_X(self.x3, self.x3_cond_x1_2, [x1_i, x2_i], 1).item()
                results['x3'].append(x3_i)

            if self.n>3:
                #sample value for x4 conditional on x1:x3
                x4_i = self.sample_x_cond_X(self.x4, self.x4_cond_x1_3, [x1_i, x2_i, x3_i], 1).item()
                results['x4'].append(x4_i)

            if self.n>4:
                #sample value for x3 conditional on x1:x2
                x5_i = self.sample_x_cond_X(self.x5, self.x5_cond_x1_4, [x1_i, x2_i, x3_i, x4_i], 1).item()
                results['x5'].append(x5_i)

            if self.n>5:
                #sample value for x3 conditional on x1:x2
                x6_i = self.sample_x_cond_X(self.x6, self.x6_cond_x1_5, [x1_i, x2_i, x3_i, x4_i, x5_i], 1).item()
                results['x6'].append(x6_i)

            if self.n>6:
                #sample value for x3 conditional on x1:x2
                x7_i = self.sample_x_cond_X(self.x7, self.x7_cond_x1_6, [x1_i, x2_i, x3_i, x4_i, x5_i, x6_i], 1).item()
                results['x7'].append(x7_i)

            if self.n>7:
                #sample value for x3 conditional on x1:x2
                x8_i = self.sample_x_cond_X(self.x8, self.x8_cond_x1_7, [x1_i, x2_i, x3_i, x4_i, x5_i, x6_i, x7_i], 1).item()
                results['x8'].append(x8_i)

            if self.n>8:
                #sample value for x3 conditional on x1:x2
                x9_i = self.sample_x_cond_X(self.x9, self.x9_cond_x1_8, [x1_i, x2_i, x3_i, x4_i, x5_i, x6_i, x7_i, x8_i], 1).item()
                results['x9'].append(x9_i)

            if self.n>9:
                #sample value for x3 conditional on x1:x2
                x10_i = self.sample_x_cond_X(self.x10, self.x10_cond_x1_9, [x1_i, x2_i, x3_i, x4_i, x5_i, x6_i, x7_i, x8_i, x9_i], 1).item()
                results['x10'].append(x10_i)

            if (i>1 and i%50==0):
                print(i, "Samples Completed")

        # format return df
        results = pd.DataFrame(results)
        results.columns = self.colnames

        return results
                
        
            
            