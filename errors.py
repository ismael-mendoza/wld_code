import subprocess
import astropy.table
import astropy.io.fits as fits
import numpy as np
from copy import deepcopy
import random
from astropy.table import Table
import os 
import matplotlib.pyplot as plt 
import fitsio 
import scipy.optimize as opt
import pickle 

import mycode.preamble as preamble

#######################################################################################################################
######################################## Errors and Bootstrap ################################################################################################################################################################################

#input np.array covariance matrix. 
def get_correlation_matrix(cov): 
    nrows,ncols = cov.shape
    corr = np.zeros((nrows, ncols))

    for i in range(nrows): 
        for j in range(i,ncols): 
            corr[i,j] = cov[i,j]/ np.sqrt(cov[i,i]*cov[j,j])
            
            #make symmetric by copying elements 
            corr[j,i] = corr[i,j]

    return corr 


#use boostrap to calculate error on the func (median, mean, etc.) 
#N is how many bootstraps to make. 
def errs_param_boot(cats, param, N, func):
    stds = []
    for cat in cats: 
        true_func = func(cat[param])
    
        #generate random N sample with replacement of func
        funcs = bootstrap_param_cats(cat, param, N, func)
            
        #stds.append((n,np.std(meds),meds)) #make it a 68% percentile instead of std, more robust. 
        print('just to check that symmetry is respected print both percentiles: {}'.format(np.absolute(np.percentile(funcs,[16,84]) - true_func)))
        stds.append(np.absolute(np.percentile(funcs,84) - true_func))
    return stds

def indp_boot_param_cats(cat, param, N, func): 
    #return a bootstrapped list of some function of parameters(median, mean, etc.) from some catalogue
    bootstrapped_list = []
    n = len(cat)
    for i in range(N):
        sample = np.random.choice(cat[param],size=n)
        bootstrapped_list.append(func(sample))
    return bootstrapped_list

#create hash map of db_id of galaxies to some parameter. 
def hash_param(cat, param): 
    hash_map = {} 
    for i in range(len(cat)): 
        hash_map[cat['db_id'][i]] = cat[param][i]
    return hash_map

#get a hash map that maps db_ids to the corresponding row in each catalogue
def hash_row(cat): 
    hash_map = {} 
    for i in range(len(cat)): 
        hash_map[cat['db_id'][i]] = cat[i]
    return hash_map

#create boostrap of a catalogue of galaxies, where we assume all of the cat in cats have the same ids and length (as in a selection cats). 
#the output is len(cats) catalogues of length N, each of the boostrapped catalogues has the same ids. 
# the output ALWAYS has the same size as the original catalogue(s). 
#The input ALWAYS has to have distinct, unique ids .
def bootstrap_cats(cats): 
    ids = list(set(cats[0]['db_id'])) #the set(.) fnc should have no effect here. 
    n = len(ids)
    assert (n == len(cats[0])), 'The cats used have repeated ids.' #make sure our cats are well formed. 

    id_sample = np.random.choice(ids,size=n) #sample some choice of ids with replacement.
    hash_maps = [hash_row(cat) for cat in cats]
    boot_rows =[[hash_maps[i][db_id] for db_id in id_sample] for i in range(len(cats))]

    return [Table(rows=rows,names=rows[0].colnames) for rows in boot_rows]

def bootstrap_param_cats(cats, param, N, func): 
    """
    - Return a bootstrapped list of some function of parameters(median, mean, etc.) from some catalogue
    this function is designed to work when cats is the output of the selection filter (so that all the cats have the 
    same galaxies per their galaxy id, this is not technically necessary but makes things easier overall). 
    You bootstrap the galaxy ids and then use that to obtain sample of func. 
    - You should pass in as one argument the 0 shear cat just for clarity. 
    - Recall the main reason for doing this is so each id_sample is repeated across N catalogues. 
    - NOTE UPDATE: ids can be repeated but the 9 cats have all the same galaxies (so each 'db_id' list is identical)
    """
    
    ids = list(cats[0]['db_id'])
    bootstrapped_params = [[] for i in range(len(cats))]
    n = len(ids)
    hash_maps = [hash_param(cat, param) for cat in cats] #get a hash map of param for cats for quick look up 

    # for each cat produce a bootstrapped sample of N funcs 
    for i in range(N):
        id_sample = np.random.choice(ids,size=n)
        for j, cat in enumerate(cats): 
            bootstrapped_params[j].append(func([hash_maps[j][gal_id] for gal_id in id_sample]))
    return bootstrapped_params

#this function takes in main_cats which are the output of the selection filter. 
#default N is 10,000 
def get_boostrap_covariance_matrix(main_cats, gi, N): 
    num_cats = len(main_cats)
    covariance_matrix = np.zeros((num_cats, num_cats))
    covariance_matrix_grp = np.zeros((num_cats, num_cats))
    
    #from each cat we get a boostrap cat of medians iso, grp    
    true_medians_iso = [np.median(cat['bias_{}'.format(gi)]) for cat in main_cats]
    true_medians_grp =[np.median(cat['bias_{}_grp'.format(gi)]) for cat in main_cats]
    medians_cat_iso = bootstrap_param_cats(main_cats, 'bias_{}'.format(gi), N, np.median)
    medians_cat_grp = bootstrap_param_cats(main_cats, 'bias_{}_grp'.format(gi), N, np.median)
    
    #now calculate covariance for each case. 
    
    for i in range(num_cats): 
        for j in range(num_cats)[:i+1]: 
            medians_iso_cat_i = np.array(medians_cat_iso[i]) 
            medians_iso_cat_j = np.array(medians_cat_iso[j]) 
            medians_grp_cat_i = np.array(medians_cat_grp[i])
            medians_grp_cat_j = np.array(medians_cat_grp[j])
            
            true_median_iso_i = true_medians_iso[i]
            true_median_iso_j = true_medians_iso[j]
            true_median_grp_i = true_medians_grp[i]
            true_median_grp_j = true_medians_grp[j]

            covariance_matrix[i,j] = np.sum(( medians_iso_cat_i - true_median_iso_i)*(medians_iso_cat_j - true_median_iso_j))/(N-1)
            covariance_matrix_grp[i,j] = np.sum(( medians_grp_cat_i - true_median_grp_i)*(medians_grp_cat_j - true_median_grp_j))/(N-1)
            
            #make symmetric by copying elements 
            covariance_matrix[j,i] = covariance_matrix[i,j]
            covariance_matrix_grp[j,i] = covariance_matrix_grp[i,j]
            
    #also calculate correlation matrices just for fun. 
    correlation_matrix = np.zeros((num_cats, num_cats))
    correlation_matrix_grp = np.zeros((num_cats,num_cats))
    for i in range(num_cats): 
        for j in range(num_cats)[:i+1]: 
            correlation_matrix[i,j] = covariance_matrix[i,j]/ np.sqrt(covariance_matrix[i,i]*covariance_matrix[j,j])
            correlation_matrix_grp[i,j] = covariance_matrix_grp[i,j]/ np.sqrt(covariance_matrix_grp[i,i]*covariance_matrix_grp[j,j])
            
            #make symmetric by copying elements 
            correlation_matrix[j,i] = correlation_matrix[i,j]
            correlation_matrix_grp[j,i] = correlation_matrix_grp[i,j]
            

    return covariance_matrix, covariance_matrix_grp,correlation_matrix,correlation_matrix_grp

#######################################################################################################################
##################################################  Money Plot ############################################
#######################################################################################################################


def linear_f_odr(B, x):
    return B[0]*x + B[1]

def semilinear_f_odr(B, x):
    return B[0]*x 

"""
This is the fit to use when assuming that all the points are not correlated. 
"""
def linear_fit(g1s,medians,fit_args): 
    import scipy.odr
    linear = scipy.odr.Model(fit_args['model'])
    mydata = scipy.odr.RealData(g1s, medians, sy=fit_args['errs'])
    myodr = scipy.odr.ODR(mydata, linear, beta0=[1.,2.])
    myoutput = myodr.run()
    #myoutput.pprint()
    beta0 = myoutput.beta[0]
    beta1 = myoutput.beta[1]
    
    return beta0, beta1, np.sqrt(myoutput.cov_beta[0,0]), np.sqrt(myoutput.cov_beta[1,1]), myoutput.cov_beta[0,1]/np.sqrt(myoutput.cov_beta[0,0]*myoutput.cov_beta[1,1])

def linear_f(b0, b1, x):
    return b0*x + b1

def cubic_f(b0,b1,b3, x): 
    return b0 + b1 * x + b3*(x**3)

def chi_sq_fit(g1s,medians, fit_args): 
    # print(g1s)
    x0=np.array([0,0])
    iso_or_grp = fit_args['iso_or_grp']
    cov = fit_args['cov_{}'.format(iso_or_grp)]
    inv = fit_args['inv_{}'.format(iso_or_grp)]

    result =  opt.minimize(chisqfunc,x0,args=(cov,fit_args['model'], medians, np.array(g1s), inv))

    b0,b1 = result.x

    #get covariance matrix of estimated parameters. 
    param_cov = chisq_cov(cov,np.array(g1s),inv)

    b0_error = np.sqrt(param_cov[0,0])
    b1_error = np.sqrt(param_cov[1,1])
    corr = param_cov[0,1]/np.sqrt(param_cov[0,0]*param_cov[1,1])
    
    return b0,b1, b0_error, b1_error, corr  

def chisqfunc(x0,cov, model, med_shear_bias,app_shear,inv=None): 
    #errs is a vector of errors on the median shear biases calculated using bootstrap 
    #corr is the correlation matrix calculated between the common elements of the 9 samples. 
    b0,b1 = x0[0], x0[1] 

    invcov = np.linalg.inv(cov)

    if inv: 
        invcov = inv 
    
    #now we used the mixed covariance matrix to calculate the chi-squared: 
    #model = b1 + b0*app_shear
    model_value = model(b0,b1,app_shear)
    chi2 = np.dot((med_shear_bias - model_value), np.dot(invcov, med_shear_bias - model_value))
    
    return chi2

#chi-squared function to calculate linear regression coefficients using covariance matrix. 
def chisq_cov(cov,app_shear,inv=None):
    invcov = np.linalg.inv(cov)
    
    if inv: 
        invcov = inv 
    
    #get H matrix, 
    H = np.zeros((len(app_shear), 2)) # 2 refers to b0,b1
    for i in range(len(app_shear)): 
        for j in range(2):
            if j == 0: 
                H[i,j] = app_shear[i]
            else: 
                H[i,j] = 1. 
    
    return np.linalg.inv(np.dot(H.T,np.dot(invcov,H)))

def get_line(xs,m,b): 
    return [m*x+b for x in xs]

def get_money_plot(g1s,g1_or_g2,errs_grp,errs_iso,cats,fit_procedure, fit_args,ticks1=None, labely1=None):
    plt.rc('text', usetex=True)
    figure1 = plt.figure(figsize=(20, 20))

    ################################ BLENDED  ################################################

    means = [np.mean(cat['bias_{}_grp'.format(g1_or_g2)]) for cat in cats]
    medians = [np.median(cat['bias_{}_grp'.format(g1_or_g2)]) for cat in cats]
    sigmas = [preamble.mad(cat['bias_{}_grp'.format(g1_or_g2)]) for cat in cats]
    
    # use the method describe in the page above for std of the median 
    fit_args['errs'] = errs_grp  #this line is only useful in the case of a linear fit

    ax = figure1.add_subplot(111)
    ax.errorbar(g1s,medians,yerr=errs_grp,marker='o',linestyle=' ',color='blue',capsize=3,label="\\rm Blending on" )
    
    fit_args['iso_or_grp'] = 'grp'
    beta0, beta1,beta0_err,beta1_err,beta01_corr = fit_procedure(g1s,medians, fit_args)

    #plot line, 
    x= g1s
    y = [cubic_f(beta0,beta1,g1) for g1 in x]
    #y = [beta0*g1 + beta1 for g1 in x]
    ax.plot(x,y,c='b')

    print("Results for fits of blended case: ")
    print()


    #this first set of print statements are good for debugging. 
    print('means grp:', means)
    print()
    print('medians grp:',medians)
    print()
    print('sigmas grp:',sigmas)
    print()
    print('errs grp:',errs_grp)
    print()

    print()
    print('value b0:     {:.3e}'.format(beta0))
    print('error b0:     {:.3e}'.format(beta0_err))
    print('value b1:     {:.3e}'.format(beta1) )
    print('error b1:     {:.3e}'.format(beta1_err))
    print('error correlation coefficient: {:.3e}'.format(beta01_corr))
    print()


    print() 
    print('###################################')

    ################################ UNBLENDED  ################################################

    means = [np.mean(cat['bias_{}'.format(g1_or_g2)]) for cat in cats]
    medians = [np.median(cat['bias_{}'.format(g1_or_g2)]) for cat in cats]
    sigmas = [preamble.mad(cat['bias_{}'.format(g1_or_g2)]) for cat in cats]
    # use the method describe in the page above for std of the median 
    fit_args['errs'] = errs_iso
    fit_args['iso_or_grp'] = 'iso'

    ax.errorbar(g1s,medians,yerr=errs_iso,marker='o',linestyle=' ',color='red',capsize=3, label = "\\rm Blending off")

    beta0, beta1,beta0_err,beta1_err,beta01_corr = fit_procedure(g1s,medians, fit_args)

    #plot line, 
    x = g1s
    y = [cubic_f(beta0,beta1,g1) for g1 in x]
    #y = [beta0*g1 + beta1 for g1 in x]
    ax.plot(x,y,c='r')


    #print useful debugging information. 
    print('means iso:',  means)
    print('medians iso:',medians)
    print('sigmas iso:', sigmas )
    print('errs iso:', errs_iso)
    print()

    print()
    print('value b0 (multiplicative bias):     {:.3e}'.format(beta0))
    print('error b0:     {:.3e}'.format(beta0_err))
    print('value b1 (additive bias):     {:.3e}'.format(beta1) )
    print('error b1:     {:.3e}'.format(beta1_err))
    print('error correlation coefficient: {:.3e}'.format(beta01_corr))
    print()


    ################################################################################################
    #formatting 
    
    #plt.ylim(-10,10)
    #ax1.set_xlim([-.025,.025])


    ax.tick_params(axis='both', which='major', labelsize=30)

    # ax1.yticks(size=20)
    num = g1_or_g2[1:]

    ax.set_xlabel(r'$g_{}$'.format(num),size=40)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    if labely1 != None: 
        ax.get_yaxis().get_offset_text().set_size(1)
        ax.set_ylabel(r'\rm {}'.format(labely1),size=40)
    else: 
        ax.get_yaxis().get_offset_text().set_size(40)


    ax.axhline(0,c='g')

    ax.tick_params(axis='both', size=10,width=3,which='both')


    if ticks1 != None: 
        ax.set_yticklabels(ticks1)
    
        
    ax.legend(loc='best', prop={'size':25})


"""

"""
def get_money_errors(g1s, g1_or_g2, errs_iso, errs_grp, cats, fit_procedure, fit_args): 

    ################################  UNBLENDED  ################################################
    #means = [np.mean(cat['bias_{}'.format(g1_or_g2)]) for cat in cats]
    medians = [np.median(cat['bias_{}'.format(g1_or_g2)]) for cat in cats]
    #sigmas = [preamble.mad(cat['bias_{}'.format(g1_or_g2)]) for cat in cats]
    
    # use the method describe in the page above for std of the median 
    fit_args['errs'] = errs_iso #only used in the linear fit function. 
    fit_args['iso_or_grp'] = 'iso'

    #beta0 = multiplicative bias. 
    #beta1 = multiplicative bias. 
    beta0_iso, beta1_iso, beta0_err_iso, beta1_err_iso, beta01_corr_iso = fit_procedure(g1s, medians, fit_args)

    ################################  BLENDED  ################################################
    #means = [np.mean(cat['bias_{}_grp'.format(g1_or_g2)]) for cat in cats]
    medians = [np.median(cat['bias_{}_grp'.format(g1_or_g2)]) for cat in cats]
    #sigmas = [preamble.mad(cat['bias_{}_grp'.format(g1_or_g2)]) for cat in cats]
    
    # use the method describe in the page above for std of the median 
    fit_args['errs'] = errs_grp #only used in the linear fit function. 
    fit_args['iso_or_grp'] = 'grp'

    beta0_grp, beta1_grp, beta0_err_grp, beta1_err_grp, beta01_corr_grp = fit_procedure(g1s, medians, fit_args)

    return beta0_iso, beta1_iso, beta0_err_iso, beta1_err_iso, beta01_corr_iso, beta0_grp, beta1_grp,beta0_err_grp,beta1_err_grp,beta01_corr_grp







#######################################################################################################################
#######################################################################################################################
###################################### WILL NOT BE USED.  #############################################################
#######################################################################################################################
#######################################################################################################################

# #calculate error on the slope for a linear function that goes through the origin with bootstrap. 
# def err_slope_boot(cats,param,filters): 
#     meds = [] #each of this is a tuple of 9 medians to fit
    
#     filter_cats = get_filter_cats(main_cats,filters)        
#     intersect_cats = get_intersection_cats(filter_cats)
      
#     #generate random N samples of the slope. 
#     N = 10000
#     slopes = []


# #use boostrap to calculate error on the median. 
# def errs_mean_boot(cats,param):
#     stds = []
#     for cat in cats: 
#         true_mean = np.mean(cat[param])
#         n = len(cat)
    
#         #generate random N sample with replacement 
#         N = 10000
#         means = bootstrap_param_cat(cat, param, N, np.mean)
            
#         #stds.append((n,np.std(meds),meds)) #make it a 68% percentile instead of std, more robust. 
#         print('just to check that symmetry is respected print both percentiles: {}'.format(np.absolute(np.percentile(means,[16,84]) - true_mean)))
#         stds.append(np.absolute(np.percentile(means,84) - true_mean))
#     return stds


# def errs_med_boot(cats,param):
#     stds = []
#     for cat in cats: 
#         n = len(cat)
#         true_median = np.median(cat[param])
#         N = 10000

#         median = bootstrap_param_cat(cat, param, N, np.mean)
#         #generate random N sample of medians with replacement 
#         meds = []
#         for i in range(N):
#             sample = np.random.choice(cat[param],size=n)
#             meds.append(np.median(sample))
            
#         #stds.append((n,np.std(meds),meds)) #make it a 68% percentile instead of std, more robust. 
#         print('just to check that symmetry is respected print both percentiles: {}'.format(np.absolute(np.percentile(meds,[16,84]) - true_median)))
#         stds.append(np.absolute(np.percentile(meds,84) - true_median))
#     return stds



# #chi-squared function to calculate linear regression coefficients using covariance matrix. 
# def get_mixcov(corr,errs):
#     #we obtain a new covariance matrix: 
#     mixcov = np.zeros(corr.shape) 
#     for i in range(corr.shape[0]): 
#         for j in range(corr.shape[1]): 
#             mixcov[i,j] = corr[i,j] * errs[i] * errs[j]
#             mixcov[j,i] = mixcov[i,j]
        
#     return mixcov


# #this function is an old first draft of the following bootstrap function. 
# def get_bias_covariance_matrices(main_cats,gi): 
#     #notice that this function limits the ellipticity bias grp to be between +1.5,-1.5 since mean function 
#     # would be heavily biased otherwise. 
#     num_cats = len(main_cats)
#     covariance_matrix = np.zeros((num_cats, num_cats))
#     covariance_matrix_grp = np.zeros((num_cats, num_cats))
#     for i in range(num_cats): 
#         for j in range(num_cats)[:i+1]: 
#             cat_i = main_cats[i]
#             cat_j = main_cats[j]
#             cut_cat1 = abs_cut(cat_i,'bias_{}_grp'.format(gi),1.5)
#             cut_cat2 = abs_cut(cat_j,'bias_{}_grp'.format(gi),1.5)
#             cut_intersection_cats = get_intersection_cats([cut_cat1,cut_cat2])
#             cut_intersect_cat1 = cut_intersection_cats[0]
#             cut_intersect_cat2 = cut_intersection_cats[1]
#             cut_length_intersection = float(len(cut_intersect_cat1))
#             b1_mean_grp,b2_mean_grp = np.mean(cut_intersect_cat1['bias_{}_grp'.format(gi)]),np.mean(cut_intersect_cat2['bias_{}_grp'.format(gi)])
#             b1_mean,b2_mean = np.mean(cut_intersect_cat1['bias_{}'.format(gi)]),np.mean(cut_intersect_cat2['bias_{}'.format(gi)])
#             covariance_matrix_grp[i,j] = np.sum((cut_intersect_cat1['bias_{}_grp'.format(gi)] - b1_mean_grp)*(cut_intersect_cat2['bias_{}_grp'.format(gi)] - b2_mean_grp))/(cut_length_intersection-1)
#             covariance_matrix[i,j] = np.sum((cut_intersect_cat1['bias_{}'.format(gi)] - b1_mean)*(cut_intersect_cat2['bias_{}'.format(gi)] - b2_mean))/(cut_length_intersection-1)
            
#             #make symmetric by copying elements 
#             covariance_matrix[j,i] = covariance_matrix[i,j]
#             covariance_matrix_grp[j,i] = covariance_matrix_grp[i,j]
            
#     #also calculate correlation matrices just for fun. 
#     correlation_matrix = np.zeros((num_cats, num_cats))
#     correlation_matrix_grp = np.zeros((num_cats,num_cats))
#     for i in range(num_cats): 
#         for j in range(num_cats)[:i+1]: 
#             correlation_matrix[i,j] = covariance_matrix[i,j]/ np.sqrt(covariance_matrix[i,i]*covariance_matrix[j,j])
#             correlation_matrix_grp[i,j] = covariance_matrix_grp[i,j]/ np.sqrt(covariance_matrix_grp[i,i]*covariance_matrix_grp[j,j])
            
#             #make symmetric by copying elements 
#             correlation_matrix[j,i] = correlation_matrix[i,j]
#             correlation_matrix_grp[j,i] = correlation_matrix_grp[i,j]
            

#     return covariance_matrix, covariance_matrix_grp,correlation_matrix,correlation_matrix_grp
    

#still obtains second plot but this plot hasn't been used at all. 
# def get_money_plot(g1s,gi,errs_grp,errs_iso,main_cats,fit_procedure,fit_args,ticks1=None,ticks2=None,ticks3=None,labely1=None):
#     plt.rc('text', usetex=True)

#     figure1 = plt.figure(figsize=(20, 20))
#     figure2 = plt.figure(figsize=(10, 10))

#     ####################BLENDED        
#     means = [np.mean(main_cat['bias_{}_grp'.format(gi)]) for main_cat in main_cats]
#     medians = [np.median(main_cat['bias_{}_grp'.format(gi)]) for main_cat in main_cats]
#     sigmas = [preamble.mad(main_cat['bias_{}_grp'.format(gi)]) for main_cat in main_cats]
    
#     # use the method describe in the page above for std of the median 
#     errs = errs_grp
#     fit_args['errs'] = errs_grp

#     ax1 = figure1.add_subplot(111)
#     ax2 = figure2.add_subplot(111)

#     print('means grp:', means)
#     print()
#     print('medians grp:',medians)
#     print()
#     print('sigmas grp:',sigmas)
#     print()
#     print('errs grp:',errs)
#     print()
#     ax1.errorbar(g1s,medians,yerr=errs,marker='o',linestyle=' ',color='blue',capsize=3,label="\\rm Blending on" )
    
#     fit_args['iso_or_grp'] = 'grp'
#     beta0, beta1,beta0_err,beta1_err,beta01_corr = fit_procedure(g1s,medians,fit_args)



    
    
#     #plot line, 
#     x= g1s
#     y = [beta0*g1 + beta1 for g1 in x]
#     ax1.plot(x,y,c='b')


#     ####second plot 
#     plt.figure()
#     x = g1s[4:]
#     y = []
#     yerrs= []
#     for i,median in enumerate(medians[4:]): 
#         y.append(medians[i] + medians[-(i+1)])
#         yerrs.append(np.sqrt(errs[i]**2 + errs[-(i+1)]**2))


#     ax2.errorbar(x,y,yerr=yerrs,marker='o',linestyle=' ',color='blue',capsize=3)
    
#     print 
#     print('###################################')

#     ################UNBLENDED 

#     means = [np.mean(main_cat['bias_{}'.format(gi)]) for main_cat in main_cats]
#     medians = [np.median(main_cat['bias_{}'.format(gi)]) for main_cat in main_cats]
#     sigmas = [preamble.mad(main_cat['bias_{}'.format(gi)]) for main_cat in main_cats]
#     # use the method describe in the page above for std of the median 
#     errs = errs_iso
#     fit_args['errs'] = errs_iso

#     ax1.errorbar(g1s,medians,yerr=errs,marker='o',linestyle=' ',color='red',capsize=3, label = "\\rm Blending off")

#     fit_args['iso_or_grp'] = 'iso'
#     beta0, beta1,beta0_err,beta1_err,beta01_corr = fit_procedure(g1s,medians, fit_args)

#     #plot line, 
#     x = g1s
#     y = [beta0*g1 + beta1 for g1 in x]
#     ax1.plot(x,y,c='r')


#     ####second plot 
#     x = g1s[4:]
#     y = []
#     yerrs= []
#     for i,median in enumerate(medians[4:]): 
#         y.append(medians[i] + medians[-(i+1)])
#         yerrs.append(np.sqrt(errs[i]**2 + errs[-(i+1)]**2))

#     ax2.errorbar(x,y,yerr=yerrs,marker='o',linestyle=' ',color='red',capsize=3)


#     #print useful debugging information. 

#     print('means iso:',  means)
#     print('medians iso:',medians)
#     print('sigmas iso:',sigmas )
#     print('errs iso:',errs)

#     ################################################################################################
#     #formatting 
    
#     #plt.ylim(-10,10)
#     #ax1.set_xlim([-.025,.025])


#     ax1.tick_params(axis='both', which='major', labelsize=30)
#     ax2.tick_params(axis='both', which='major', labelsize=30)


#     # ax1.yticks(size=20)
#     num = gi[1:]

#     ax1.set_xlabel(r'$g_{}$'.format(num),size=40)
#     ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
#     if labely1 != None: 
#         ax1.get_yaxis().get_offset_text().set_size(1)
#         ax1.set_ylabel(r'\rm {}'.format(labely1),size=40)
#     else: 
#         ax1.get_yaxis().get_offset_text().set_size(40)



#     ax2.set_xlabel(r'${}$'.format(gi),size=40)


#     ax1.axhline(0,c='g')
#     ax2.axhline(0,c='g')

#     ax1.tick_params(axis='both', size=10,width=3,which='both')
#     ax2.tick_params(axis='both', size=10,width=3,which='both')


#     if ticks1 != None: 
#         ax1.set_yticklabels(ticks1)
    
    
#     if ticks2 != None: 
#         ax2.set_yticklabels(ticks2)
        
#     if ticks3 != None: 
#         ax2.set_xticklabels(ticks3)
        
#     ax1.legend(loc='best', prop={'size':25})


    # print
    # print(fit_args['iso_or_grp'] + ':')
    # print
    # print('value b0:     {:.3e}'.format(beta0))
    # print('error b0:     {:.3e}'.format(np.sqrt(myoutput.cov_beta[0,0])))
    # print('std error b0: {:.3e}'.format(np.sqrt(myoutput.sd_beta[0])))

    # print('value b1:     {:.3e}'.format(beta1))
    # print('error b1:     {:.3e}'.format(np.sqrt(myoutput.cov_beta[1,1])))
    # print('std error b1: {:.3e}'.format(np.sqrt(myoutput.sd_beta[1])))
    # print 