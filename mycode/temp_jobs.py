import os 


survey_name = 'LSST'
project_name = 'project'

##step size to try: 
#0.001
#0.01
#0.03
#e.g. ss1 = step size of 1/1000 

####################################### simulate #######################################

# for g in ['-.005','-.01','-.015','-.02','0.','.005','.01','.015','.02']: 
#for g in ['-0.05','-0.10']: 
# for g in ['-.02','-.015', '-.01', '-.005','.005','.01','.015', '0.02']: 
#for g in [ '0.02']:
#for g in ['0.0']: 
#for g in ['-0.1','-0.05','-.02','-.015', '-.01', '-.005','.005','.01','.015', '0.02', '0.05', '0.1']: 
    # os.system(f'python mycode/all-process.py --simulate-all --num-sections 12 --cosmic-shear-g1 {g} --cosmic-shear-g2 0 --project {project_name}{survey_name}-g1_{int(float(g)*1000)}-g2_0_ss1 --survey-name {survey_name}')



####################################### the rest #######################################

for g in ['.02']:
#for g in ['-.005','-.01','-.015','-.02','0.','.005','.01','.015']:  
#for g in ['-.02','-.015', '-.01', '-.005','.005','.01','.015', '0.02']: 
# for g in ['-0.1','-0.05','-.02','-.015', '-.01', '-.005','0.','.005','.01','.015', '0.02', '0.05', '0.1']: 
# for g in ['0.0']: 
# for g in ['-0.1','-0.05','-.02','-.015', '-.01', '-.005','.005','.01','.015', '0.02', '0.05', '0.1']: 
    # os.system('bsub -W 30:00 -o "output-{2}{1}-g1_{0}-g2_0.txt" -r "python all-process.py --process-all --add-noise-all --extract-all --combine  --num-sections 10 --project {2}{1}-g1_{0}-g2_0_ss1 --noise-seed 0 --survey-name {1}" '.format(int(float(g)*1000),survey_name,project_name))
    os.system('bsub -W 30:00 -o "output-{2}{1}-g1_{0}-g2_0_ss1.txt" -r "python mycode/all-process.py --add-noise-all --extract-all --combine  --num-sections 12 --project {2}{1}-g1_{0}-g2_0_ss1 --noise-seed 0 --survey-name {1}" '.format(int(float(g)*1000),survey_name,project_name))










####################################### process  #######################################
# for g in ['-0.1','-0.05','-.02','-.015', '-.01', '-.005','0.','.005','.01','.015', '0.02', '0.05', '0.1']: 
# for g in ['0.0']: 
#     os.system('bsub -W 30:00 -o "output-{2}{1}-g1_{0}-g2_0.txt" -r "python all-process.py --process-all --num-sections 10 --project {2}{1}-g1_{0}-g2_0 --noise-seed 0 --survey-name {1}" '.format(int(float(g)*1000),survey_name, project_name))


####################################### add noise #######################################


####################################### extract #######################################

####################################### combine #######################################

########### new noise addition:
# project = 'fproject2'
# for g in ['-.005','-.01','-.015','-.02','0.','.005','.01','.015','.02']:  
# for g in ['-0.1','-0.05','-.02','-.015', '-.01', '-.005','0.','.005','.01','.015', '0.02', '0.05', '0.1']: 

    # os.system('mkdir {2}{1}-g1_{0}-g2_0'.format(int(float(g)*1000),survey_name,project))
    # for i in range(10):
    #     for j in range(10):
    #         old_project =  '{}{}-g1_{}-g2_0'.format('project', 'LSST', int(float(g)*1000))
    #         curr_project = '{}{}-g1_{}-g2_0'.format(project, 'LSST', int(float(g)*1000))
    #         file_name = '{}/{}{}{}.fits'.format(old_project,'section',i,j)
    #         os.system('cp {} {}'.format(file_name,curr_project))

    
   # os.system('bsub -W 30:00 -o "output-{2}{1}-g1_{0}-g2_0.txt" -r "python all-process.py --add-noise-all --extract-all --combine --num-sections 10 --project {2}{1}-g1_{0}-g2_0 --noise-seed 123 --survey-name {1}" '.format(int(float(g)*1000),survey_name,project))


# for g in ['.02']: 
#     os.system('bsub -W 30:00 -o "output-projectLSST-g1_{0}-g2_0.txt" -r "python all-process.py --extract-all --combine --num-sections 10 --project projectLSST-g1_{0}-g2_0 --noise-seed 0 --survey LSST" '.format(int(float(g)*1000)))


# for g in ['0.']: 
#     os.system('bsub -W 30:00 -o "output-projectLSST-g1_{0}.txt" -r "python all-process.py --process-all --add-noise-all --extract-all --combine --num-sections 10 --project projectLSST-g1_{0} --noise-seed 0 --survey LSST" '.format(int(float(g)*1000)))






# #########single image simulation that Javier suggested. 
# project = 'projectd2'
# # os.system('python all-process.py --simulate-single --single-image-width 4096 --single-image-height 4096 --single-image-ra-center .4 --single-image-dec-center -.2 --cosmic-shear-g1 0. --cosmic-shear-g2 0. --project {} --survey-name {}'.format(project, survey_name))
# os.system('bsub -W 30:00 -o "output-projectd2.txt" -r "python all-process2.py --project {} --survey-name {} --process-single --add-noise-single --extract-single --noise-seed 123" '.format(project,survey_name))

