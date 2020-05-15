import numpy as np
import scipy.optimize as opt
from . import weights


# input np.array covariance matrix.
def get_correlation_matrix(cov):
    nrows, ncols = cov.shape
    corr = np.zeros((nrows, ncols))

    for i in range(nrows):
        for j in range(i, ncols):
            corr[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])

            # make symmetric by copying elements
            corr[j, i] = corr[i, j]

    return corr


def get_boostrap_covariance_matrix(orig_ids, main_cats, fnc, N=1000, args_iso=[], args_grp=[], return_bootstraps=False):
    """
    This function takes in main_cats which are the output of the selection filter, default N is 10,000
    """
    num_cats = len(main_cats)
    covariance_matrix = np.zeros((num_cats, num_cats))
    covariance_matrix_grp = np.zeros((num_cats, num_cats))

    # from each cat we get a bootstrap cat of medians iso, grp
    true_values_iso = [fnc(orig_ids, cat, *args_iso) for cat in main_cats]
    true_values_grp = [fnc(orig_ids, cat, *args_grp) for cat in main_cats]
    values_cat_iso = weights.boots_fnc(orig_ids, main_cats, fnc, N=N, args=args_iso)
    values_cat_grp = weights.boots_fnc(orig_ids, main_cats, fnc, N=N, args=args_grp)

    # now calculate covariance for each case.
    for i in range(num_cats):
        for j in range(num_cats)[:i + 1]:
            values_iso_cat_i = np.array(values_cat_iso[i])
            values_iso_cat_j = np.array(values_cat_iso[j])
            values_grp_cat_i = np.array(values_cat_grp[i])
            values_grp_cat_j = np.array(values_cat_grp[j])

            true_value_iso_i = true_values_iso[i]
            true_value_iso_j = true_values_iso[j]
            true_value_grp_i = true_values_grp[i]
            true_value_grp_j = true_values_grp[j]

            covariance_matrix[i, j] = np.sum(
                (values_iso_cat_i - true_value_iso_i) * (values_iso_cat_j - true_value_iso_j)) / (N - 1)
            covariance_matrix_grp[i, j] = np.sum(
                (values_grp_cat_i - true_value_grp_i) * (values_grp_cat_j - true_value_grp_j)) / (N - 1)

            # make symmetric by copying elements
            covariance_matrix[j, i] = covariance_matrix[i, j]
            covariance_matrix_grp[j, i] = covariance_matrix_grp[i, j]

    if not return_bootstraps:
        return covariance_matrix, covariance_matrix_grp, get_correlation_matrix(
            covariance_matrix), get_correlation_matrix(covariance_matrix_grp)
    else:
        return covariance_matrix, covariance_matrix_grp, get_correlation_matrix(
            covariance_matrix), get_correlation_matrix(covariance_matrix_grp), values_cat_iso, values_cat_grp


def linear_f(b, x):
    return b[0] * x + b[1]


def cubic_f(b, x):
    return b[0] + b[1] * x + b[2] * (x ** 3)


def penta_f(b, x):
    return b[0] + b[1] * x + b[2] * (x ** 3) + b[3] * (x ** 5)


def get_line(xs, m, b):
    return m * xs + b


def chi_sq_fit(g1s, ys, inv, model):
    """
    Model is linear_f or cubic_f for example. 
    """

    if model is linear_f:
        x0 = np.array([0, 0])

    elif model is cubic_f:
        x0 = np.array([0, 0, 0])

    elif model is penta_f:
        x0 = np.array([0, 0, 0, 0])

    else:
        raise NotImplementedError("Have not yet implemented that model.")

    result = opt.minimize(chisqfunc, x0, args=(inv, model, ys, np.array(g1s)))

    b = result.x

    # get covariance matrix of estimated parameters.
    param_cov = chisq_cov(inv, np.array(g1s))

    # b0_error = np.sqrt(param_cov[0,0])
    # b1_error = np.sqrt(param_cov[1,1])
    # corr = param_cov[0,1]/np.sqrt(param_cov[0,0]*param_cov[1,1])

    return b, param_cov


def chisqfunc(x0, inv, model, ys, g1s):
    # errs is a vector of errors on the median shear biases calculated using bootstrap
    # corr is the correlation matrix calculated between the common elements of the 9 samples.

    # now we used the mixed covariance matrix to calculate the chi-squared:
    # model = b1 + b0*app_shear
    model_value = model(x0, g1s)
    chi2 = np.dot((ys - model_value), np.dot(inv, ys - model_value))

    return chi2


# chi-squared function to calculate linear regression coefficients using covariance matrix.
def chisq_cov(inv, g1s):
    H = np.zeros((len(g1s), 2))  # 2 refers to b0,b1
    for i in range(len(g1s)):
        for j in range(2):
            if j == 0:
                H[i, j] = g1s[i]
            else:
                H[i, j] = 1.

    return np.linalg.inv(np.dot(H.T, np.dot(inv, H)))


###############################################################################################################
###############################################################################################################

# def extract_betas(betas, cov, model):
#     if model is linear_f: 
#         beta0_err, beta1_err = np.sqrt(np.diag(cov))
#         beta01_corr = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
#         return (betas[0], betas[1], beta0_err, beta1_err, beta01_corr)

#     elif model is cubic_f: 
#         # beta0_err, beta1_err, beta2_err = np.sqrt(np.diag(cov))
#         return (betas[0], betas[1], betas[2])

def prepare_money_plot(g1s, orig_ids, scats, fnc, fit_procedure=chi_sq_fit, N=1000, model=linear_f, args_iso=[],
                       args_grp=[]):
    assert fit_procedure == chi_sq_fit, "Not yet implemented..."
    assert model == linear_f or model == cubic_f, "What are you trying to do?, not yet implemented"

    values = [fnc(orig_ids, cat, *args_iso) for cat in scats]
    values_grp = [fnc(orig_ids, cat, *args_grp) for cat in scats]

    cov, cov_grp, corr, corr_grp, boot_values, boot_values_grp = get_boostrap_covariance_matrix(orig_ids, scats, fnc,
                                                                                                N=N, args_iso=args_iso,
                                                                                                args_grp=args_grp,
                                                                                                return_bootstraps=True)

    errs = np.sqrt(cov.diagonal())
    errs_grp = np.sqrt(cov_grp.diagonal())

    inv, inv_grp = np.linalg.inv(cov), np.linalg.inv(cov_grp)

    return (values, values_grp), (cov, cov_grp, inv, inv_grp, corr, corr_grp), (errs, errs_grp), (
    boot_values, boot_values_grp)


def make_money_plot(g1s, values, errs, values_grp, errs_grp, inv, inv_grp, fit_procedure=chi_sq_fit, model=linear_f,
                    ticks1=None, labely1=None, ax=None, colors=None, extra=None,
                    legend_size=25, tick_size=40, label_size=40):
    """
    * The errors are obtained as the squaroot of the diagonal of the covariance matrix. 
    * betas = [betas_iso, betas_grp] in that order. 
    * For simplicity this function only supports money plot for g1 component on bias. 
    """

    plt.rc('text', usetex=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 20), nrows=1, ncols=1)

    if colors is None:
        colors = ['red', 'blue']

    ax.errorbar(g1s, values, yerr=errs, marker='o', linestyle=' ', color=colors[0], capsize=3,
                label="\\rm Blending off{}".format(extra))
    ax.errorbar(g1s, values_grp, yerr=errs_grp, marker='o', linestyle=' ', color=colors[1], capsize=3,
                label="\\rm Blending on{}".format(extra))

    # plot line
    x = g1s
    betas, param_cov = fit_procedure(g1s, values, inv, model)
    betas_grp, param_cov_grp = fit_procedure(g1s, values_grp, inv_grp, model)

    if model is linear_f:
        beta0, beta1 = betas
        (beta0_err, beta1_err), beta01_corr = np.sqrt(np.diag(param_cov)), param_cov[0, 1] / np.sqrt(
            param_cov[0, 0] * param_cov[1, 1])
        beta0_grp, beta1_grp = betas_grp
        (beta0_err_grp, beta1_err_grp), beta01_corr_grp = np.sqrt(np.diag(param_cov_grp)), param_cov_grp[
            0, 1] / np.sqrt(param_cov_grp[0, 0] * param_cov_grp[1, 1])

        y = linear_f(betas, x)
        ax.plot(x, y, c=colors[0])

        y_grp = linear_f(betas_grp, x)
        ax.plot(x, y_grp, c=colors[1])

        print("Results for fits of unblended case: \n")

        print('\n value b0:     {:.3e}'.format(beta0))
        print('error b0:     {:.3e}'.format(beta0_err))
        print('value b1:     {:.3e}'.format(beta1))
        print('error b1:     {:.3e}'.format(beta1_err))
        print('error correlation coefficient: {:.3e}\n'.format(beta01_corr))

        print("Results for fits of blended case: \n")

        print('\n value b0:     {:.3e}'.format(beta0_grp))
        print('error b0:     {:.3e}'.format(beta0_err_grp))
        print('value b1:     {:.3e}'.format(beta1_grp))
        print('error b1:     {:.3e}'.format(beta1_err_grp))
        print('error correlation coefficient: {:.3e}'.format(beta01_corr_grp))

    elif model is cubic_f or model is penta_f:
        y = model(betas, x)
        ax.plot(x, y, c=colors[0])

        y_grp = model(betas_grp, x)
        ax.plot(x, y_grp, c=colors[1])

    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    ax.set_xlabel(r'$g_1$', size=label_size)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    if labely1 != None:
        ax.get_yaxis().get_offset_text().set_size(1)
        ax.set_ylabel('\\rm {}'.format(labely1), size=label_size)
    else:
        ax.get_yaxis().get_offset_text().set_size(40)

    ax.axhline(0, linestyle='--', color='r')

    ax.tick_params(axis='both', size=10, width=3, which='both')

    if ticks1 != None:
        ax.set_yticklabels(ticks1)

    ax.legend(loc='best', prop={'size': legend_size})
