#this file should be run inside the WLD directory. 
import subprocess 
import os 

project_name = "project1"
boot_name = "boot1"
selection_cats_loc = 'mycode/selection_cats1.p' #relative to WLD folder. 
shears_num = 9 
tomo_num = 6
tomos_py_dir = 'mycode/tomo/tomos.py' #relative to WLD folder. 

# subprocess.run(f"python {tomo_py_dir} --setup --tomo-num {tomo_num} --project {project_name} --shears-num {shears_num} --selection-cats {selection_cats_loc} --N2 10000", shell=True)

#cmd = f"python {tomos_py_dir} --boot --tomo-num {tomo_num} --project {project_name} --shears-num {shears_num} --selection-cats {selection_cats_loc} --N1 10000 --N2 10000 --boot-dir {boot_name} --first-boot-num 0"


#cmd = f"python {tomos_py_dir} --correlations-prep --project {project_name} --shears-num {shears_num} --tomo-num {tomo_num} --selection-cats {selection_cats_loc} --boot-dir {boot_name}"

cmd = f"python {tomos_py_dir} --correlations-run --project {project_name} --shears-num {shears_num} --tomo-num {tomo_num} --selection-cats {selection_cats_loc} --boot-dir {boot_name} --result-boot-num 5373"



print()
print(cmd)
subprocess.run(cmd, shell=True)
#subprocess.run(f'''bsub -W 20:00 -o "mycode/tomo/project1/correlations-output.txt" -r "{cmd}" ''', shell=True)


# zero_cat_number = 4
