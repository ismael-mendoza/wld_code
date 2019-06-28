import numpy as np

#param is usually one of the biases. 
def get_iso_or_grp_suffix(iso_or_grp):
    suffix = ''
    if iso_or_grp == 'grp': 
        suffix = '_grp'
        
    return suffix 
    

def get_weights(args, component, iso_or_grp): 
    suffix = get_iso_or_grp_suffix(iso_or_grp)
    shape_noise = np.std(args[f'e{component}'])**2 
    weights = (shape_noise + args[f'dg{component}{suffix}']**2)**(-1)
    
    return weights 

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


def wmean_func(dbt, cat, component, iso_or_grp):
    assert component in ['1','2'], 'component value is not valid. '
    assert iso_or_grp in ['iso', 'grp'], 'iso_or_grp received invalied argument'
    
    suffix = get_iso_or_grp_suffix(iso_or_grp)
    weights = get_weights(args, component, iso_or_grp)
    return np.sum(weights*dbt)/np.sum(weights) 

def wmean(dbt, args, component, iso_or_grp): 
    return lambda dbt: wmean_func(dbt, args, component, iso_or_grp)


def get_error(dbt, fnc, N=1000): 
    results = boot_fnc(dbt, fnc, N)
    return np.std(results)

def boot_fnc(dbt, fnc, N=1000):
    results = [] 
    for i in range(N): 
        results.append(fnc(np.random.choice(dbt,len(dbt), replace=True)))
    return results 

def clipped_mean(dbt, p): 
    if p == 0.5: 
        return np.median(dbt)
    
    q1 = np.quantile(dbt, p)
    q2 = np.quantile(dbt, 1-p)
    cut_dbt = cut_both(dbt, q1, q2)
    mean = np.mean(cut_dbt)
    return mean 

def clipped_mean_fnc(p): 
    return lambda dbt: clipped_mean(dbt, p) 

def cut_both(dbt, cut1, cut2): 
    return dbt[(dbt>=cut1)&(dbt<=cut2)]