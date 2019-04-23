#should be ran from inside WLD folder. 
import os 
import pickle
import subprocess 
import argparse
from mycode.tomo.tomo_fncs import * 
from mycode.errors import * 

wld_dir = "/nfs/slac/g/ki/ki19/deuce/AEGIS/ismael/WeakLensingDeblending"
tomo_dir = os.path.join(wld_dir, "mycode/tomo")

#required arguments. 
parser = argparse.ArgumentParser(description='Run single extraction of tomos.')
parser.add_argument('--project', type=str, required = True, help="Where is the project for this tomography analysis?, relative to the tomo folder.")
parser.add_argument('--tomo-num', type=int, required=True, help="Number of tomo bins between 0 and 1.2, the last bin will from 1.2 and on. Default should be 6.")
parser.add_argument('--shears-num', type=int, help="How many shears used?", required = True)
parser.add_argument('--selection-cats', type=str, required = True, 
                     help="Where is the selection cat to be use as a base for this project relative to WLD?")


parser.add_argument('--setup', action='store_true', help="Copy selection cats and create project directory to get ready to run bootstrap. Also run analysis in the original selection cats")
parser.add_argument('--boot', action='store_true', help="Run bootstrap analysis fo the tomos to find correlation between multilicative biases.")
parser.add_argument('--correlations-prep', action='store_true', help="Extract all the multiplicative biases calculated in boot folder into one file.")
parser.add_argument('--correlations-run', action='store_true', help="Calculate correlations from the multiplicative biases that were bootstrapped with the --boot command, this command should be used after running the --correlations-prep.")
parser.add_argument('--fit', action='store_true', help="Perform a fit of the multiplicative biases using the calculated correlations.")


parser.add_argument('--boot-dir', type=str, help="Where relative to the project folder should we save all the bootstraps? relative to project folder.")
parser.add_argument('--N1', type=int, help="How many samples in the outer bootstrap?", default=None)
parser.add_argument('--N2', type=int, help="How many samples in the inner bootstrap?", default=None)
parser.add_argument('--first-boot-num', type=int, help="What should we number the first boot result?", default=0)
parser.add_argument('--result-boot-num', type=int, help="How many boots went into obtaining the multiplicative biases", default=None)


#parse the arguments into args. 
args = parser.parse_args()


project_dir = os.path.join(tomo_dir, args.project)

#store the original selection cat name and the new location in tomodir. Also checked we did no go wrong. 
selection_cats_name = os.path.basename(args.selection_cats)
assert os.path.basename(selection_cats_name) != '', "You included a slash in the name of selection cat!"

selection_cats_loc = os.path.join(project_dir, selection_cats_name)


if args.setup: 

    if not os.path.isdir(project_dir):
        os.mkdir(project_dir)

    selection_cats_old_loc = os.path.join(wld_dir,args.selection_cats)

    #copy selection cat to the new project directory. 
    subprocess.run(f"cp {selection_cats_old_loc} {project_dir}", shell=True)

    #run analysis on selection cat inside project_dir with no extra string. 
    # subprocess.run(f"python onetomo.py --project {project_dir} --selection-cats {selection_cats_loc} --zero-cat-number {args.zero_cat_number} --N {args.N2} --shears-num {args.shears_num}", shell=True)

    cmd_to_bsub = f"python {os.path.join(tomo_dir,'one_tomo.py')} --project {project_dir} --selection-cats {selection_cats_loc} --N {args.N2} --shears-num {args.shears_num} --tomo-num {args.tomo_num}"

    subprocess.run(f'''bsub -W 20:00 -o "{os.path.join(tomo_dir,'output_orig_tomo.txt')}" -r "{cmd_to_bsub}" ''', shell = True) #triple quotes!


if args.boot: 
    assert args.N1 is not None and args.N2 is not None and args.boot_dir is not None, "You are missing --N1, --N2, or --boot-dir arguments." 

    project_boot_dir = os.path.join(project_dir, args.boot_dir)
    
    #make directory to store bootstraps if it does not exist yet. 
    if not os.path.isdir(project_boot_dir):
        os.mkdir(project_boot_dir)


    #the following loop will send a job for each desired outer bootstrap. 
    for i in range(args.first_boot_num, args.N1 + args.first_boot_num): 
        #save this bootstrapped instance of selection cat. 
        boot_cat_loc = os.path.join(project_boot_dir,f"boot_cats{i}.p")

        #run the one tomo in the newly bootstrapped selection cat through slac. 
        # subprocess.run(f'bsub -W 20:00 -o "output_boot_{i}.txt" -r "python mycode/tomo/onetomo.py --project {project_boot_dir} --selection-cats {boot_cat_loc} --zero-cat-number {args.zero_cat_number} --N {args.N2} --shears-num {args.shears_num} --tomo-num {args.tomo_num} --extra-str {i} "', shell = True)

        cmd_to_bsub = f"python {os.path.join(tomo_dir,'one_tomo.py')} --project {project_boot_dir} --selection-cats {selection_cats_loc} --boot-cats {boot_cat_loc} --N {args.N2} --shears-num {args.shears_num} --tomo-num {args.tomo_num} --extra-str {i}"

        #print(cmd_to_bsub)
        subprocess.run(f'''bsub -W 20:00 -o "{os.path.join(project_boot_dir,f'output_boot{i}.txt')}" -r "{cmd_to_bsub}" ''', shell = True) 


# --boot must have been used previously.  
if args.correlations_prep: 
    assert args.boot_dir is not None, "You are missing --boot-dir arguments." 

    #dump all multiplicative biases into a list for both iso and grp. One list for each tomo. 
    #each list inside will have all the bootstrapped ms corresponding to the tomo number. 
    all_boot_ms_iso = [[] for x in range(args.tomo_num)] 
    all_boot_ms_grp = [[] for x in range(args.tomo_num)] 

    project_boot_dir = os.path.join(project_dir, args.boot_dir)

    files_in_boot = os.listdir(project_boot_dir)
    ms_files_in_boot = [s for s in files_in_boot if 'tomo_ms' in s]  #obtain all the files in the boot dir folder that have the mss. 
    for file in ms_files_in_boot: 
        #boot_tomo_cats, boot_tomo_errs, boot_tomo_bootstrap_matrices, boot_tomo_ms = load_tomos(tomo_boot_dir, extra_str=f'{i}')
        boot_tomo_ms = pickle.load(open(os.path.join(project_boot_dir, file),'rb'),encoding='latin1')
        boot_tomo_ms_iso, boot_tomo_ms_errs_iso, boot_tomo_ms_grp, boot_tomo_ms_errs_grp = boot_tomo_ms

        for tomo_n in range(args.tomo_num): 
            all_boot_ms_iso[tomo_n].append(boot_tomo_ms_iso[tomo_n])
            all_boot_ms_grp[tomo_n].append(boot_tomo_ms_grp[tomo_n])
    
    n_ms_files = len(ms_files_in_boot)
    print(f'The number of boots current in the boot dir are: {n_ms_files}')

    pickle.dump(all_boot_ms_iso, open(os.path.join(project_dir,f'all_boot_ms_iso{n_ms_files}.p'),'wb'))
    pickle.dump(all_boot_ms_grp, open(os.path.join(project_dir,f'all_boot_ms_grp{n_ms_files}.p'),'wb'))
    #read true catalogue to obtain those with real ms. 

if args.correlations_run: 
    assert args.boot_dir is not None and args.result_boot_num is not None, "You are missing --boot-dir or --result-boot-num arguments."


    all_boot_ms_iso = pickle.load(open(os.path.join(project_dir,f'all_boot_ms_iso{args.result_boot_num}.p'),'rb'), encoding='latin1')
    all_boot_ms_grp = pickle.load(open(os.path.join(project_dir,f'all_boot_ms_grp{args.result_boot_num}.p'),'rb'), encoding='latin1')

    cov_tomo_ms_iso = np.zeros((args.tomo_num,args.tomo_num))
    cov_tomo_ms_grp = np.zeros((args.tomo_num, args.tomo_num))


    tomo_ms = pickle.load(open(os.path.join(project_dir, 'tomo_ms.p'),'rb'),encoding='latin1')
    (tomo_ms_iso, tomo_ms_errs_iso, tomo_ms_grp, tomo_ms_errs_grp) = tomo_ms 


    #calculate covariance matrix. 
    for i in range(args.tomo_num): 
        for j in range(i,args.tomo_num):
            for k in range(args.result_boot_num):
                cov_tomo_ms_iso[i,j] += (all_boot_ms_iso[i][k] - tomo_ms_iso[i]) * (all_boot_ms_iso[j][k] - tomo_ms_iso[j])
                cov_tomo_ms_grp[i,j] += (all_boot_ms_grp[i][k] - tomo_ms_grp[i]) * (all_boot_ms_grp[j][k] - tomo_ms_grp[j])
            cov_tomo_ms_iso[j,i] = cov_tomo_ms_iso[i,j]
            cov_tomo_ms_grp[j,i] = cov_tomo_ms_grp[i,j]

    #divide by number of samples per covariance matrix definition. 
    cov_tomo_ms_iso =  (1/(args.result_boot_num - 1)) * cov_tomo_ms_iso
    cov_tomo_ms_grp =  (1/(args.result_boot_num - 1)) * cov_tomo_ms_grp

    #get correlation matrices. 
    corr_tomo_ms_iso = get_correlation_matrix(cov_tomo_ms_iso)
    corr_tomo_ms_grp = get_correlation_matrix(cov_tomo_ms_grp)

    #pickle and we are done. 
    pickle.dump((cov_tomo_ms_iso, cov_tomo_ms_grp, corr_tomo_ms_iso, corr_tomo_ms_grp), open(os.path.join(project_dir, f"cov_corr_tomo_ms{args.result_boot_num}.p"),'wb'))

    print(f'Successfully saved the covariance matrices produced with {args.result_boot_num} bootstraps.')





#parser.add_argument('--zero-cat-number', type=int, required=True, help="For the 9/n catalogues for each shear, which one is the one with zero shear?")




# #directories to be used. 

# tomo_dir = 
# boot_dir = os.path.join(tomo_dir,"boot")


# #make if not yet there. 

# if not os.path.isdir(boot_dir):
#     os.mkdir(boot_dir)

# #save everything to tomo_dir
# #get selection cats with no ambiguous blends etc from pickle. 


# #obtain the bootstrap matrix for each set of catalogues. 

# #tomos = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]



# #save_tomo_cats(tomo_cats, tomo_dir) #this save 

# #pickle both multiplicate thingies. 

# #now bootstrap fun time. We are going to run each of the N1 iterations in SLAC.  
# N1 = 10000
# N2 = 10000
# for j in range(N1): 

#     pickle.dump(tomo_boot_cats, os.path.join(boot_dir,f"tomo_cats{j}.p" )) 

#     #boostrap from selection cat. 

# tomographic_bootstrap_matrices = [] 
# for i,cats in enumerate(tomographic_cats): 
    
#     #get boostrapped errors. 
#     errs_grp = errs_param_boot(cats,'bias_g1_grp', np.median) 
#     errs_iso = errs_param_boot(cats,'bias_g1', np.median) 
#     tomographic_errs_grp.append(errs_grp)
#     tomographic_errs_iso.append(errs_iso)
    
#     #get boostrapped matrices. 
#     zero_cat = cats[4] #catalogue with zero shear. 
 


#     tomographic_bootstrap_matrices.append((boot_covariance_matrix, boot_covariance_matrix_grp, boot_correlation_matrix, boot_correlation_matrix_grp))
    

