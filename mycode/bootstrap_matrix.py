"""
The objective of this file is to get a catalogue as pickle file as an input and calculate the bootstrapped covariance matrices as explained in the paper. Then save this matrices as a pickle file. 

This process can potentially take a long time so this is why we want it as a separate file (to run in slac or non-locally).
"""
import errors 
import preamble 
import argparse 


parser = argparse.ArgumentParser(description='Run bootstrap matrix procedure on pickled catalogues')
parser.add_argument('--cat', type=str, required=True)
parser.add_argument('--zero-cat', type=str, required=True)
parser.add_argument('--result', type=str, required=True)
args = parser.parse_args()


#extract catalogue pickle file. 
cat = pickle.load( open(f"{args.cat}.p", "rb" ), enconding='latin1')
zero_cat = pickle.load( open( f"{args.zero_cat}.p", "rb" ), enconding='latin1')


boot_covariance_matrix, boot_covariance_matrix_grp, boot_correlation_matrix, boot_correlation_matrix_grp = errors.get_boostrap_covariance_matrix(cat,'g1',zero_cat) 

#picked obtain matrices. 
pickle.dump((boot_covariance_matrix, boot_covariance_matrix_grp, boot_correlation_matrix, boot_correlation_matrix_grp),open(f"{args.result}.p","wb"))