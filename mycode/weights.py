import numpy as np



####################################################################################################################
####################################################################################################################


################## utility ############################
def cut_both(dbt, cut1, cut2): 
    return dbt[(dbt>=cut1)&(dbt<=cut2)]

#param is usually one of the biases. 
def get_iso_or_grp_suffix(iso_or_grp):
    suffix = ''
    if iso_or_grp == 'grp': 
        suffix = '_grp'
        
    return suffix 

############################ boostrap ###################################

def boots_fnc(orig_ids, cats, fnc, N=1000, args=[]):
    """
    All cats must have same ids and sorted by db_id. (orig_ids =[0,1,2,..., n-1])
    """
    bootstrap_ids = [np.random.choice(orig_ids,len(orig_ids), replace=True) for _ in range(N)] 
    results = [[fnc(ids, cat, *args) for ids in bootstrap_ids] for cat in cats]  
    return results 


######################## medians #####################
def median_fnc(ids, cat, component, iso_or_grp): 
    suffix= get_iso_or_grp_suffix(iso_or_grp)
    return np.median(cat[ids][f'bias_g{component}{suffix}'])


############################ weighted means. ############################
    

def get_weights(ids, cat, component, iso_or_grp, which_shape_noise='component', shape_noise=None): 
    suffix = get_iso_or_grp_suffix(iso_or_grp)
    
    if shape_noise is None: 
        if which_shape_noise=='component': 
            shape_noise = np.std(cat[f'e{component}'][ids])**2
        elif which_shape_noise=='magnitude': 
            shape_noise = np.std(np.sqrt(cat['e1'][ids]**2 + cat['e2'][ids]**2))**2
        
        
    weights = (shape_noise + cat[f'dg{component}{suffix}'][ids]**2)**(-1)
    
    return weights 

#which_shape_noise in []'component','magnitude'
def wmean(ids, cat, component, iso_or_grp, which_shape_noise, shape_noise=None):
    assert component in ['1','2'], 'component value is not valid. '
    assert iso_or_grp in ['iso', 'grp'], 'iso_or_grp received invalied argument'
    assert which_shape_noise in ['component', 'magnitude'], 'invalid agument for which_shape_noise'
    
    suffix = get_iso_or_grp_suffix(iso_or_grp)
    weights = get_weights(ids, cat, component, iso_or_grp, which_shape_noise, shape_noise)
    param = f'bias_g{component}{suffix}'
    dbt = cat[param][ids]
    return np.sum(weights*dbt)/np.sum(weights)


#all catalogues must be same size (so ids make sense) and sorted. 
def get_errors(orig_ids, cats, fnc, N=1000, args=[]): 
    results = boots_fnc(orig_ids, cats, fnc, N, args)
    return [np.std(cat_results) for cat_results in results]



######################### clippings means ###################################


def clipped_mean(ids, cat, p, param): 
    dbt = cat[param][ids]
    return clipped_mean_simple(dbt, p)

def clipped_mean_simple(dbt, p):
    
    if p == 0.5: 
        return np.median(dbt)

    q1 = np.quantile(dbt, p)
    q2 = np.quantile(dbt, 1-p)
    cut_dbt = cut_both(dbt, q1, q2)
    mean = np.mean(cut_dbt)
    return mean 


####################### clipped weighted mean #######################

def clipped_weighted_mean(ids, cat, p, param, component, iso_or_grp, which_shape_noise):
    """
    In this function, shape noise is always calculated from original catalogue. 
    """

    #clip. 
    temp = cat[ids]
    dbt = temp[param]
    q1 = np.quantile(dbt, p)
    q2 = np.quantile(dbt, 1-p)
    final = temp[ (dbt>=q1)&(dbt<=q2)]
    fids = list(range(len(final)))

    shape_noise =  np.std(cat[f'e{component}'])**2
    return wmean(fids, final, component, iso_or_grp, which_shape_noise, shape_noise)


######################## misc. #####################


def get_not_dropped_cat(cat): 
    grps = np.sort(list(set(cat['grp_id']))) #ids of all the groups, sorted. 
    
    #create hash map from grp_id to index in catalogue 
    hash_grpid = {} 
    for i, row in enumerate(cat): 
        grp_id = row['grp_id']
        if hash_grpid.get(grp_id, None) is None: 
            hash_grpid[grp_id] = [] 
        hash_grpid[grp_id].append(i)
    
    grp_sizes = [cat[hash_grpid[grp_id][0]]['grp_size'] for grp_id in grps] #in order of grps. 
    
    grp_dropped = []  #in order of grps. 
    for grp_id in grps: 
        cnt = 0
        for idx in hash_grpid[grp_id]:
            if cat[idx]['snr_grpf'] == 0: 
                cnt+=1
        grp_dropped.append(cnt)
        
    grps_not_dropped = np.array(grps)[np.array(grp_dropped) == 0]
    return get_slice(cat, 'grp_id', set(list(grps_not_dropped)))





# def boots_fnc_simple(dbt, fnc, N=1000):
#     results = [] 
#     for i in range(N): 
#         results.append(fnc(np.random.choice(dbt,len(dbt), replace=True)))
#     return results 
