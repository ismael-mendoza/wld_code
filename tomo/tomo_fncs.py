# this should be ran from inside the WLD folder. 
import pickle
import os 
from mycode.preamble import *
import mycode.errors as errors
import numpy as np 


#input: (selected) catalogues for different shears. 
#output: a tomogrpahic binning of each catalogue using Pat's suggestion which is 5 bins between (0.2, 1.2)
# for z and one overflow bin >1.2. The result is a list of lists of cats for each shear and tomography 
def get_tomographic_cats(cats, tomos): 
    tomo_cats = [] 
    
    for i in range(len(tomos)-1) : 
        tomo_cats.append([])
        for cat in cats: 
            z_1 = tomos[i]
            z_2 = tomos[i+1]
            bin_cat = down_cut(up_cut(cat,'z', z_1), 'z',z_2)
            tomo_cats[i].append(bin_cat)
    
    #put in the last bin with everything that has z > 1.2 
    tomo_cats.append([])
    for cat in cats: 
        bin_cat = up_cut(cat, 'z',tomos[-1])
        tomo_cats[-1].append(bin_cat)

    return tomo_cats
    

def get_tomo_errs(tomo_cats, N): 
    tomo_errs_iso = []
    tomo_errs_grp = [] 
    for i,cats in enumerate(tomo_cats): 
        #get boostrapped errors. 
        errs_iso = errors.errs_param_boot(cats,'bias_g1', N, np.median) 
        errs_grp = errors.errs_param_boot(cats,'bias_g1_grp', N, np.median) 
        tomo_errs_iso.append(errs_iso)
        tomo_errs_grp.append(errs_grp)

    return (tomo_errs_iso, tomo_errs_grp)

"""
a
"""
def get_tomo_bootstrap_matrices(tomo_cats, N): 

    tomo_bootstrap_matrices = [] 
    for i,cats in enumerate(tomo_cats): 

        boot_covariance_matrix, boot_covariance_matrix_grp, boot_correlation_matrix, boot_correlation_matrix_grp = errors.get_boostrap_covariance_matrix(cats,'g1', N)  

        tomo_bootstrap_matrices.append((boot_covariance_matrix, boot_covariance_matrix_grp, boot_correlation_matrix, boot_correlation_matrix_grp))  

    return tomo_bootstrap_matrices


def get_tomo_multp_bias(g1s, tomo_cats, tomo_errs, tomo_bootstrap_matrices):
    tomo_ms_iso = [] 
    tomo_ms_grp = [] 
    tomo_ms_errs_iso = [] 
    tomo_ms_errs_grp = []
    for i,(cats, errs_iso, errs_grp, (boot_covariance_matrix, boot_covariance_matrix_grp, boot_correlation_matrix, boot_correlation_matrix_grp)) in enumerate(zip(tomo_cats, *tomo_errs, tomo_bootstrap_matrices)):

        beta0_iso, beta1_iso,beta0_err_iso,beta1_err_iso,beta01_corr_iso, beta0_grp, beta1_grp,beta0_err_grp,beta1_err_grp,beta01_corr_grp = errors.get_money_errors(g1s, 'g1', errs_iso, errs_grp, cats, errors.chi_sq_fit, {'cov_iso':boot_covariance_matrix, 'cov_grp':boot_covariance_matrix_grp,'inv_iso':None,'inv_grp':None, 'model':errors.linear_f})

        tomo_ms_iso.append(beta0_iso)
        tomo_ms_grp.append(beta0_grp)
        tomo_ms_errs_iso.append(beta0_err_iso)
        tomo_ms_errs_grp.append(beta0_err_grp)

    return (tomo_ms_iso, tomo_ms_errs_iso, tomo_ms_grp, tomo_ms_errs_grp)

#here we might need to do parallel computing. 
def get_tomo_multp_correlations(tomo_cats, ms): 
    raise NotImplementedError 
    # get 10000 multiplicative biases for each tomo cat. 



def save_tomos(tomo_errs, tomo_bootstrap_matrices, tomo_ms, tomo_dir, extra_str=''):
    pickle.dump(tomo_errs, open(os.path.join(tomo_dir, f"tomo_errs{extra_str}.p"),'wb'))
    pickle.dump(tomo_bootstrap_matrices, open(os.path.join(tomo_dir,f"tomo_bootstrap_matrices{extra_str}.p"), 'wb'))
    pickle.dump(tomo_ms, open(os.path.join(tomo_dir,f"tomo_ms{extra_str}.p"), 'wb'))


def load_tomos(project_dir, extra_str =''):
    # tomo_cats = pickle.load(open(os.path.join(project_dir, f'tomo_cats{extra_str}.p'),'rb'), encoding='latin1')
    tomo_errs= pickle.load(open(os.path.join(project_dir, f"tomo_errs{extra_str}.p"),'rb'), encoding='latin1')
    tomo_bootstrap_matrices = pickle.load(open(os.path.join(project_dir, f"tomo_bootstrap_matrices{extra_str}.p"),'rb'), encoding='latin1')

    tomo_ms = pickle.load(open(os.path.join(project_dir, f"tomo_ms{extra_str}.p"),'rb'), encoding='latin1')

    return tomo_errs, tomo_bootstrap_matrices, tomo_ms





# tomo_ms_errs = pickle.load(open(os.path.join(tomo_dir, f"tomo_ms_errs{extra_str}.p"),'rb'), encoding='latin1')

# pickle.dump((tomo_errs_grp, tomo_errs_iso) , )

# pickle.dump(tomo_bootstrap_matrices, os.path.join(tomo_dir,))
# pickle.dump(tomo_ms, os.path.join(tomo_dir,))
# pickle.dump(tomo_ms_errs, os.path.join(tomo_dir,f"tomo_ms_errs{extra_str}.p"))

# """
# Save tomographic cats so that they can be fed in the bias_matrix thingy. 
# """
# def save_tomo_cats(tomo_cats, tomo_dir): 
#     for i,tomo_cat in enumerate(tomo_cats):
#         pickle.dump(tomo_cat, os.path.join(tomo_dir, f"tomo_cat{i}.p"))
