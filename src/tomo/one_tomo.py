#this should be ran from inside the WLD folder. 
import argparse 
import numpy as np 
from src.tomo.tomo_fncs import *
from mycode.errors import bootstrap_cats


parser = argparse.ArgumentParser(description='Run single tomographic analysis.')
parser.add_argument('--project', type=str, required=True, help='Absolute path to project or boot folder')
parser.add_argument('--selection-cats', type=str, required=True, help='Absolute path to original selection cats.')
parser.add_argument('--boot-cats', type=str, required=True, help='New location/name of bootstrap cat to create. ')
parser.add_argument('--tomo-num', type=int, required=True, help='Default is 6.')
parser.add_argument('--extra-str',type=str, default='', help="Use to save files with a different ending.")
parser.add_argument('--N', type=int, required=True, 
                    help='How many samples to use for the bootstraps to calculate covariance matrices and errors.')
parser.add_argument('--shears-num', type=int, help="How many shears used?, default is 9.", required = True)
args = parser.parse_args()

tomos = np.linspace(0.2, 1.2, args.tomo_num)

if args.shears_num == 9: 
    g1s = [-.02,-.015,-.01,-.005,0.,.005,.01,.015,.02]
elif args.shears_num == 13: 
    g1s = [-.1, -.05, -.02,-.015,-.01,-.005,0.,.005,.01,.015,.02, .05, .1]
else: 
    raise ValueError("Shear number specified is not available.")


#read selection cats specified. 
print("Reading previous selection cats...")
selection_cats = pickle.load(open(args.selection_cats, 'rb'), encoding='latin1')


#obtain one boostrap from the read selection cat. 
#each cat in selection_cats must have the same length. 
#each cat in 'bootstrap_selection_cats' will have the same length as each cat in 'selection_cats'. 
print("Obtaining new bootstrap catalogue...")
bootstrap_selection_cats = bootstrap_cats(selection_cats) #each time you get a different one.  

#pickle.dump(bootstrap_selection_cats, open(args.boot_cats, 'wb'))

#get tomo_cats
tomo_cats = get_tomographic_cats(bootstrap_selection_cats, tomos)

#get median errors for each catalogue in each tomo. 
print("getting errors...")
tomo_errs = get_tomo_errs(tomo_cats, args.N)

#obtain bootstrap matrices. 
print("getting bootstrap matrices...")
tomo_bootstrap_matrices = get_tomo_bootstrap_matrices(tomo_cats, args.N)

#obtain multiplicative , additive errors. 
print("getting multiplicative biases...")
tomo_ms = get_tomo_multp_bias(g1s, tomo_cats, tomo_errs, tomo_bootstrap_matrices)


#save them all into specified project directory.
print("saving tomo_errs,tomo_bootstrap_matrices, tomo_ms only...")
save_tomos(tomo_errs, tomo_bootstrap_matrices, tomo_ms, args.project, extra_str=args.extra_str)