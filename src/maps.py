import random
from collections import defaultdict
from copy import deepcopy

import numpy as np


dflt_sorting = ['db_id']


# get hash map from param to list of indices
def get_hash(cat, param):
    hash_map = defaultdict(list)
    for i, row in enumerate(cat):
        hash_map[row[param]].append(i)
    return hash_map


# in order of grps.
def get_group_sizes(cat, grps):
    hash_grpid = get_hash(cat, 'grp_id')
    return [cat[hash_grpid[grp_id][0]]['grp_size'] for grp_id in grps]


def get_non_duplicated_cat(cat):
    """
    * Get catalogue with no duplicated copies, this is necessary because some of them might be in more than one patch. 

    * We further drop the whole group with this duplicated galaxies because its hard to decide which one to keep and its contributoin would appear in two separate groups, etc. 

    * assume cat is sorted according to dflt_sorting above.
    """
    keep = [True] * len(cat)  # initially we keep everything.

    dbid_hash = get_hash(cat, 'db_id')
    grpid_hash = get_hash(cat, 'grp_id')

    for db_id, idxs in dbid_hash.items():
        if len(idxs) >= 2:  # duplicated
            for idx in idxs:
                grp_id = cat[idx]['grp_id']
                grp_idxs = grpid_hash[grp_id]
                for i in grp_idxs:
                    keep[i] = False
    return cat[keep]


# smaller random samples
def leaveRandom(sz, cat):
    new_cat = deepcopy(cat)
    rm_rows = random.sample(range(0, len(cat) - 1), len(cat) - sz)
    new_cat.remove_rows(rm_rows)
    return new_cat


def get_filter_cats(cats, filters):
    filter_cats = []
    for c in cats:
        cat = deepcopy(c)
        for fil in filters:
            cat = fil(cat)
        filter_cats.append(cat)
    return filter_cats


# if shear_zero_cat is none then assume is the middle one from the ones passed by.
# use shear = 0 as the galaxies to be selected across samples to avoid selection bias
def selection_filter(cats, filters, zero_shear_cat):
    filter_cats = []

    zero_cat = deepcopy(zero_shear_cat)

    for fil in filters:
        zero_cat = fil(zero_cat)

    zero_ids = set(zero_cat['db_id'])

    # in this step, remember we DO NOT apply the filter for the other cats that are not the zero shear cat,
    # we just select the same galaxies found in the filtered zero shear cat. 
    for cat in cats:
        filter_cats.append(get_slice(cat, 'db_id', zero_ids))

    icats = get_intersection_cats(filter_cats)  # shouldn't do anything.
    fcats = []
    for cat in icats:  # sort them.
        cat.sort(dflt_sorting)
        fcats.append(cat)

    return fcats  # make sure all ids are the same.


def get_slice(cat, field, selection, non_selection=False):
    """
    this function returns the rows of cat that are in selection. 
    """
    selection = set(list(selection))  # make sure it's a set.
    bool_slice = []
    for data in cat[field]:
        bool_slice.append(data in selection)

    if not non_selection:
        return cat[bool_slice]
    else:
        return cat[bool_slice], cat[np.invert(bool_slice)]


def get_intersection_cats(cats, other_cats=False):
    """
    gets the ids of the galaxies that is contained in each cat in cats and returns a new list new_cats 
    that contains only this galaxies
    """
    intersection = set()
    for i, cat in enumerate(cats):
        if i == 0:
            intersection = set(cat['db_id'])
        intersection = intersection.intersection(cat['db_id'])

    # convert intersection ids to indices , then remove indices that are not those ones:
    new_cats = []
    new_non_cats = []
    for cat in cats:
        new_cat, new_non_cat = get_slice(cat, 'db_id', intersection, non_selection=True)
        new_cats.append(new_cat)
        new_non_cats.append(new_non_cat)

    if other_cats:
        return new_cats, new_non_cats
    else:
        return new_cats


def get_fish_matrices(fits_file, grp_ids, descwl, only_fish=False):
    matrices = []
    reader = descwl.output.Reader(fits_file)
    results = reader.results
    for grp_id in grp_ids:
        selected = results.select(f'grp_id=={grp_id}')
        sort_order = np.argsort(results.table['grp_rank'][selected])
        selected = selected[sort_order]
        fisher, cov, var, corr = results.get_matrices(selected)
        if only_fish:
            matrices.append(fisher)
        else:
            matrices.append(((fisher, cov, var, corr), selected))
    return matrices


def get_not_dropped_cat(cat):
    grps = cat['grp_id']

    # create hash map from grp_id to index in catalogue
    hash_grpid = {}
    for i, grp_id in enumerate(grps):
        if hash_grpid.get(grp_id, None) is None:
            hash_grpid[grp_id] = []
        hash_grpid[grp_id].append(i)

    grp_dropped = []
    for grp_id in grps:
        cnt = 0
        for idx in hash_grpid[grp_id]:
            if cat[idx]['snr_grpf'] == 0:
                cnt += 1
        grp_dropped.append(cnt)

    return cat[np.array(grp_dropped) == 0]

    # grps_not_dropped = np.array(grps)[np.array(grp_dropped) == 0]
    # return get_slice(cat, 'grp_id', set(list(grps_not_dropped)))


# some interesting subsets of the simulation
iso_gal = lambda cat: cat[cat['purity'] > .98]  # isolated galaxies
grp_gal = lambda cat: cat[cat['purity'] <= .98]  # galaxies in a group of 2 or more.

# 'good' galaxies satisfy the reasonable criteria below.
good = lambda cat: cat[(cat['snr_grpf'] > 6) & (cat['sigma_m'] > .2)]

# gold sample galaxies
gold = lambda cat: cat[(cat['ab_mag'] < 25.3)]

# ambiguity of blends.
ambig = lambda cat: cat[cat['ambig_blend'] == True]
not_ambig = lambda cat: cat[cat['ambig_blend'] == False]
detected = lambda cat: cat[cat['match'] != -1]
not_detected = lambda cat: cat[cat['match'] == -1]

# cuts
cut_biasiso = lambda cat, bias_cut: cat[
    (np.absolute(cat['bias_g1']) < bias_cut) & (np.absolute(cat['bias_g2']) < bias_cut)]
cut_biasgrp = lambda cat, bias_cut: cat[
    (np.absolute(cat['bias_g1_grp']) < bias_cut) & (np.absolute(cat['bias_g2_grp']) < bias_cut)]
down_cut = lambda cat, param, cut: cat[cat[param] < cut]
up_cut = lambda cat, param, cut: cat[cat[param] > cut]
abs_cut = lambda cat, param, cut, point=0: cat[np.absolute(cat[param] - point) < cut]
unphysical_iso = lambda cat: cat[(np.absolute(cat['bias_g1']) > 1.) | (abs(cat['bias_g2']) > 1.)]
unphysical_grp = lambda cat: cat[(np.absolute(cat['bias_g1_grp']) > 1.) | (np.absolute(cat['bias_g2_grp']) > 1.)]

# more specific
detc_and_notambig = lambda cat: cat[(cat['ambig_blend'] == False) & (cat['match'] != -1)]
notdetc_and_notambig = lambda cat: cat[(cat['ambig_blend'] == False) & (cat['match'] == -1)]
detc_and_ambig = lambda cat: cat[(cat['ambig_blend'] == True) & (cat['match'] != -1)]
notdetc_and_ambig = lambda cat: cat[(cat['ambig_blend'] == True) & (cat['match'] == -1)]
best = detc_and_notambig

low_cond = lambda cat: cat[cat['cond_num_grp'] < 1e14]
not_dropped = lambda cat: cat[cat['snr_grpf'] > 0]

# filter rare (bad) objs. Which will not be good for our purposes.
not_bad = lambda cat: cat[(cat['snr_grp'] != 0) & (cat['ds_grp'] != np.inf)]

# get a smaller sample of each of the catalogs for illustrative purposes
small = lambda cat, N: leaveRandom(N, cat)
