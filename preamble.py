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
import os 

#directories that would be using 
locations = dict(
mycode = '/nfs/slac/g/ki/ki19/deuce/AEGIS/ismael/WeakLensingDeblending/mycode', 
wld = '/nfs/slac/g/ki/ki19/deuce/AEGIS/ismael/WeakLensingDeblending',
aegis = '/nfs/slac/g/ki/ki19/deuce/AEGIS/ismael',
)


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def plot_matrix(matrix, param_names): 

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    n,_ = matrix.shape #assume square matrix
    plt.xticks(list(plt.xticks()[0]), [None] + param_names)
    plt.yticks(list(plt.yticks()[0]), [None] + param_names)
    
    plt.tick_params(labelsize=20)


    ax.matshow(matrix, cmap=plt.cm.Blues)


    for i in range(n):
        for j in range(n):
            c = matrix[j,i]
            ax.text(i, j, '{:.2g}'.format(c), va='center', ha='center',size=20)

## Prepare input 

files = dict(
final_fitsLSST1 = os.path.join(locations['wld'], 'projectLSST-g1_-100-g2_0/final_fits.fits'), 
final_fitsLSST2 = os.path.join(locations['wld'], 'projectLSST-g1_-50-g2_0/final_fits.fits'), 
final_fitsLSST3 = os.path.join(locations['wld'], 'projectLSST-g1_-20-g2_0/final_fits.fits'), 
final_fitsLSST4 = os.path.join(locations['wld'], 'projectLSST-g1_-15-g2_0/final_fits.fits'), 
final_fitsLSST5 = os.path.join(locations['wld'], 'projectLSST-g1_-10-g2_0/final_fits.fits'), 
final_fitsLSST6 = os.path.join(locations['wld'], 'projectLSST-g1_-5-g2_0/final_fits.fits'), 
final_fitsLSST7 = os.path.join(locations['wld'], 'projectLSST-g1_0-g2_0/final_fits.fits'), 
final_fitsLSST8 = os.path.join(locations['wld'], 'projectLSST-g1_5-g2_0/final_fits.fits'), 
final_fitsLSST9 = os.path.join(locations['wld'], 'projectLSST-g1_10-g2_0/final_fits.fits'), 
final_fitsLSST10 = os.path.join(locations['wld'], 'projectLSST-g1_15-g2_0/final_fits.fits'), 
final_fitsLSST11 = os.path.join(locations['wld'], 'projectLSST-g1_20-g2_0/final_fits.fits'), 
final_fitsLSST12 = os.path.join(locations['wld'], 'projectLSST-g1_50-g2_0/final_fits.fits'), 
final_fitsLSST13 = os.path.join(locations['wld'], 'projectLSST-g1_100-g2_0/final_fits.fits'), 
    

final_fitsLSST1_ss1 = os.path.join(locations['wld'],'projectLSST-g1_-20-g2_0_ss1/final_fits.fits'),
final_fitsLSST2_ss1 = os.path.join(locations['wld'],'projectLSST-g1_-15-g2_0_ss1/final_fits.fits'),
final_fitsLSST3_ss1 = os.path.join(locations['wld'],'projectLSST-g1_-10-g2_0_ss1/final_fits.fits'),
final_fitsLSST4_ss1 = os.path.join(locations['wld'],'projectLSST-g1_-5-g2_0_ss1/final_fits.fits'),
final_fitsLSST5_ss1 = os.path.join(locations['wld'],'projectLSST-g1_0-g2_0_ss1/final_fits.fits'),
final_fitsLSST6_ss1 = os.path.join(locations['wld'],'projectLSST-g1_5-g2_0_ss1/final_fits.fits'),
final_fitsLSST7_ss1 = os.path.join(locations['wld'],'projectLSST-g1_10-g2_0_ss1/final_fits.fits'),
final_fitsLSST8_ss1 = os.path.join(locations['wld'],'projectLSST-g1_15-g2_0_ss1/final_fits.fits'),
final_fitsLSST9_ss1 = os.path.join(locations['wld'],'projectLSST-g1_20-g2_0_ss1/final_fits.fits'),
)

## Samples 

#some interesting subsets of the simulation
iso_gal = lambda cat: cat[cat['purity'] > .98] #isolated galaxies
grp_gal = lambda cat: cat[cat['purity'] <= .98] #galaxies in a group of 2 or more. 

#'good' galaxies satisfy the reasonable criteria below.
good = lambda cat: cat[(cat['snr_grpf'] > 6) & (cat['sigma_m'] > .2)]

#gold sample galaxies 
gold = lambda cat: cat[(cat['ab_mag'] < 25.3)] 

#ambiguity of blends. 
ambig = lambda cat: cat[cat['ambig_blend'] == True ]
not_ambig = lambda cat: cat[cat['ambig_blend'] == False ]
detected = lambda cat: cat[cat['match'] != -1]
not_detected = lambda cat: cat[cat['match'] == -1]

#cuts 
cut_biasiso = lambda cat,bias_cut: cat[(np.absolute(cat['bias_g1']) < bias_cut) & (np.absolute(cat['bias_g2']) < bias_cut)]
cut_biasgrp = lambda cat,bias_cut: cat[(np.absolute(cat['bias_g1_grp']) < bias_cut) & (np.absolute(cat['bias_g2_grp']) < bias_cut)]
down_cut = lambda cat,param,cut: cat[cat[param] < cut]
up_cut = lambda cat,param,cut: cat[cat[param] > cut]
abs_cut = lambda cat,param,cut,point=0: cat[np.absolute(cat[param] - point) < cut]
unphysical_iso = lambda cat: cat[(np.absolute(cat['bias_g1']) > 1.) | (abs(cat['bias_g2']) > 1.)]
unphysical_grp = lambda cat: cat[(np.absolute(cat['bias_g1_grp']) > 1.) | (np.absolute(cat['bias_g2_grp']) > 1.)]

#more specific 
detc_and_notambig = lambda cat: cat[(cat['ambig_blend'] == False) & (cat['match'] != -1)]
notdetc_and_notambig = lambda cat: cat[(cat['ambig_blend'] == False) & (cat['match'] == -1)]
detc_and_ambig = lambda cat: cat[(cat['ambig_blend'] == True) & (cat['match'] != -1)]
notdetc_and_ambig = lambda cat: cat[(cat['ambig_blend'] == True) & (cat['match'] == -1)]
best = detc_and_notambig

#filter rare (bad) objs. Which will not be good for our purposes. 
not_bad = lambda cat: cat[(cat['snr_grp'] != 0) & (cat['ds_grp']!= np.inf) ]

def get_non_duplicated_cat(cat):
    #get non-duplicated and duplicated objects
    seen = set()
    duplic = set()
    a = list(cat['db_id'])

    for x in a:
        if x in seen: 
            duplic.add(x)
        if x not in seen:
            seen.add(x)

    #get duplicated and non-duplicated objects
    duplic_cat, non_duplic_cat = get_slice(cat, 'db_id', duplic,non_selection='True')    
    return non_duplic_cat

#smaller random samples 
def leaveRandom(sz,cat):
    new_cat = deepcopy(cat)
    rm_rows = random.sample(range(0,len(cat)-1), len(cat) - sz)
    new_cat.remove_rows(rm_rows)
    return new_cat 

#get a smaller sample of each of the catalogs for illustrative purposes 
small = lambda cat,N: leaveRandom(N, cat)

def get_filter_cats(cats,filters): 
    filter_cats = []
    for c in cats: 
        cat = deepcopy(c)
        for fil in filters:
            cat = fil(cat) 
        filter_cats.append(cat)
    return filter_cats

#if shear_zero_cat is none then assume is the middle one from the ones passed by. 
#use shear = 0 as the galaxies to be selected across samples to avoid selection bias 
def selection_filter(cats,filters,zero_shear_cat): 
    filter_cats = []

    zero_cat = deepcopy(zero_shear_cat)

    for fil in filters: 
        zero_cat = fil(zero_cat)
        
    zero_ids = set(zero_cat['db_id'])
    
    #in this step, remember we DO NOT apply the filter for the other cats that are not the zero shear cat, 
    # we just select the same galaxies found in the filtered zero shear cat. 
    for cat in cats: 
        filter_cats.append(get_slice(cat, 'db_id', zero_ids))
        
    return get_intersection_cats(filter_cats) #double make sure all ids are the same. 

#this function returns the rows of cat that are in selection. 
def get_slice(cat, field, selection,non_selection=False): 
    bool_slice = []
    for data in cat[field]: 
        bool_slice.append(data in selection)
        
    if not non_selection: 
        return cat[bool_slice]
    else: 
        return cat[bool_slice], cat[np.invert(bool_slice)]

#gets the ids of the galaxies that is contained in each cat in cats and returns a new list new_cats 
#that contains only this galaxies
def get_intersection_cats(cats,other_cats=False): 
    intersection = set() 
    for i,cat in enumerate(cats): 
        if i==0: 
            intersection = set(cat['db_id']) 
        intersection = intersection.intersection(cat['db_id'])
            
    #convert intersection ids to indices , then remove indices that are not those ones:
    new_cats = []
    new_non_cats = [] 
    for cat in cats: 
        new_cat, new_non_cat = get_slice(cat, 'db_id', intersection, non_selection=True)
        new_cats.append(new_cat)
        new_non_cats.append(new_non_cat)
        
    if other_cats: 
        return new_cats,new_non_cats
    else: 
        return new_cats




# files_temps = dict() 
# for f in files: 
#     l = files[f].split("/")
#     temp_file = '{0}{1}'.format('/Users/Ismael/temp_data/',l[-1])
#     files_temps[f] = temp_file

    
# files_slac = dict()
# for f in files: 
#     l = files[f].split("/")
#     index_data = l.index("aegis")
#     str_slac = "/".join(l[index_data:])
#     slac_file = '{0}{1}'.format('/nfs/slac/g/ki/ki19/deuce/AEGIS/ismael/',str_slac)
#     files_slac[f] = slac_file