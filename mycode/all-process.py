#####This file is to be run when rooted in WLD. 

import numpy as np 
import os
import sys 
import argparse
import subprocess
import fitsio
import astropy 
import astropy.io.fits as fits
import astropy.table as astroTable 
from WeakLensingDeblending import descwl 
from loguru import logger 

@logger.catch
def detected_ambiguous_blends(table,matched_indices,detected):
    """
    Algorithm for ambiguous blends - returns indices of ambiguously blended objects in terms of the table of full catalogue. 

    Ambiguously blended means that at least one non-detected true object is less than a unit of effective distance away from an ambiguous object. 

    table: table containing entries of all galaxies of the catalog. 
    matched_indices: indices of galaxies in table that were detected by SExtractor (primary  matched)
    detected: Table of detected objects by SExtractor which also contains their information as measured by SExtractor. 
    """
    ambiguous_blends_indices = []
    for index,gal_row in enumerate(table):
        if gal_row['match'] == -1: #loop through true not primarily matched objects. 
            
            #calculate of effective distances of true non-primary match object with all primarily matched objects. 
            effective_distances = list(np.sqrt((detected['X_IMAGE']-gal_row['dx'])**2 + (detected['Y_IMAGE']-gal_row['dy'])**2)/(detected['SIGMA'] + gal_row['psf_sigm']))
            
            #mark all objects with effective distance <1 as ambiguosly blended. 
            marked_indices = [matched_indices[i] for i,distance in enumerate(effective_distances) if distance < 1.]
            
            #if at least one index was marked as ambiguous, add both the primarily match object and the true non-primary object as ambiguous. 
            if len(marked_indices) > 0:
                ambiguous_blends_indices.append(index)
                ambiguous_blends_indices.extend(marked_indices)
    return set(ambiguous_blends_indices)

@logger.catch
def main():

    #names to be used.
    WLD_dir = '/nfs/slac/g/ki/ki19/deuce/AEGIS/ismael/WLD'
    data_dir = '/nfs/slac/g/ki/ki19/deuce/AEGIS/ismael/WLD/data'
    logs_dir = f'{WLD_dir}/logs'
    params_dir=f"{WLD_dir}/params"

    #setup logger. 
    logger.add(f"{logs_dir}/all-process.log", format="{time}-{level}: {message}", level="INFO", backtrace=True, rotation="7:00", enqueue=True) #new file created every day at 7:00 am . 

    #setup argparser. 
    parser = argparse.ArgumentParser(description=('Simulate different regions from a square' 
                                                  'degree and analyze their combination in'
                                                  'SExtractor.'),
                                     formatter_class=(
                                     argparse.ArgumentDefaultsHelpFormatter))

    parser.add_argument('--simulate-all', action='store_true',
                        help=('Simulates the requested regions with given job_number using multiple batch jobs.'))

    parser.add_argument('--add-noise-all', action='store_true',
                        help=('Add noise to all of the images (one for each section) of the project.'))

    parser.add_argument('--extract-all', action='store_true',
                        help=('Classifies into detected and ambiously blended for a one square degree.'))

    parser.add_argument('--combine', action='store_true',
                        help=('Combines regions with given job_number'))

    # parser.add_argument('--process-all', action='store_true',
    #                     help=('Prepare files needed to run pipeline in one square degree .'))


    parser.add_argument('--simulate-single', action='store_true',
                        help=('Simulate a single chip with width,height specified with single-image-height,etc.'))

    # parser.add_argument('--process-single', action='store_true',
    #                     help=('Prepare files needed to run pipeline in a single chip.'))

    parser.add_argument('--extract-single', action='store_true',
                        help=('Classifies into detected and ambiously blended for a single section.'))

    parser.add_argument('--add-noise-single', action='store_true',
                        help=('Create a noise image for an image .fits file in current project. Have to pass in the name of the image to add noise to.'))

    parser.add_argument('--single-image-width', default=None,
                        type=int,
                        help=('Width of single image to be simulated (in pixels).'))

    parser.add_argument('--single-image-height', default=None,
                        type=int,
                        help=('Height of single image to be simulated (in pixels).'))

    parser.add_argument('--single-image-ra-center', default=0.0,
                        type=float,
                        help=('Ra center of the single image to be simulated'))

    parser.add_argument('--single-image-dec-center', default=0.0,
                        type=float,
                        help=('Dec center of the single image to be simulated'))

    parser.add_argument('--section-name', default='section',
                        type=str,
                        help=('specify name of the section image when it is a single one.'))


    parser.add_argument('--bjob-time', default='10',
                        type=str,
                        help=('Time of jobs to be run in SLAC'))

    parser.add_argument('--max-memory', default='4096',
                        type=str,
                        help=('Max memory to be used when running a slac process.'))

    parser.add_argument('--cosmic-shear-g1', default=None,
                        type=float,
                        help=('cosmic shear g1'))

    parser.add_argument('--cosmic-shear-g2', default=None, 
                        type=float,
                        help=('cosmic shear g2'))

    parser.add_argument('--noise-seed', default=None,
                        type=int,
                        help=('noise seed to use when adding noise to image SExtracted.'))

    parser.add_argument('--survey-name', required=True, 
                        type=str, choices=['LSST','HSC','DES'],
                        help=('Select survey to use.'))

    parser.add_argument('--num-sections', default=None, 
                        type=int,
                        help=('Number of squares along one dimension to divide the one square degree.'))


    parser.add_argument('--project', required=True,
                        type=str,
                        help=('Project name, directory name for all your files.'))

    parser.add_argument('--noise-name', default='noise_image',
                        type=str,
                        help=('specify name of the noise iamge'))

    parser.add_argument('--outcat-name', default='outcat',
                        type=str,
                        help=('specify name of the output catalogue from sextracting.'))

    parser.add_argument('--final-name', default='final_fits',
                        type=str,
                        help=('specify name of the final fits'))

    parser.add_argument('--table-name', default='table',
                        type=str,
                        help=('specify name of the table'))


    args = parser.parse_args()

    #make sure only arguments that make sense are selected. 
    if (args.simulate_all or args.simulate_single) and (args.cosmic_shear_g1==None or args.cosmic_shear_g2==None): 
        raise RuntimeError('Need to include the cosmic shear when simulating.')

    if (args.simulate_all or args.combine or args.process_all or args.add_noise_all or args.extract_all) and not args.num_sections: 
        raise RuntimeError('Need to include number of sections when dealing with whole one square degree.')

    if (args.add_noise_single or args.add_noise_all) and args.noise_seed==None: 
        raise RuntimeError('To add noise you need the noise seed!')

    if args.simulate_single and (args.single_image_height==None or args.single_image_width==None): 
        raise RuntimeError('Need size of the image when processing a single section, default is 4096.')


    ###################################################################
    #some constants used throughout

    if args.survey_name == 'LSST': 
        pixel_scale = .2
    elif args.survey_name == 'HSC': 
        pixel_scale = 0.17
    elif args.survey_name == 'DES': 
        pixel_scale = .263

    logger.info(f"\n Using survey {args.survey_name} and pixel scale {pixel_scale}")
    SECTION_NAME = 'section'

    ###################################################################

    logger.info(f"Remember by default we assume we are running in SLAC cluster with WLD directory being: {WLD_dir}")

    inputs = dict(
    project = f'{data_dir}/{args.project}', 
    noise_image = f'{data_dir}/{args.project}/{args.noise_name}.fits',
    final_fits = f'{data_dir}/{args.project}/{args.final_name}.fits',
    single_section = f'{data_dir}/{args.project}/{args.section_name}.fits',
    output_detected = f'{data_dir}/{args.project}/{args.outcat_name}.cat',
    config_file = f'{params_dir}/sextractor_runs/default.sex',
    param_file = f'{params_dir}/sextractor_runs/default.param',
    filter_file = f'{params_dir}/sextractor_runs/default.conv', 
    starnnw_file = f'{params_dir}/sextractor_runs/default.nnw',
    WLD = WLD_dir,
    simulate_file = f'{WLD_dir}/WeakLensingDeblending/simulate.py',
    sample_fits = f'{params_dir}/example.fits',
    one_sq_degree = f'{params_dir}/OneDegSq.fits',
    )

    #create project directory if does not exist yet
    if not os.path.exists(inputs['project']):
        os.makedirs(inputs['project'])

    os.chdir(inputs['WLD'])  #convenience so we can run simulate... 


    ######################################################################################################################################################################################################################################
    #simulate the regions. 


    if args.simulate_all:
        #constants used for the endpoints. 
        endpoint2 = (1.-1./args.num_sections)/2 #in degrees. 
        endpoint1 = -endpoint2
        total_height = 1. * 60 * 60 / pixel_scale #in pixels
        total_width = 1. * 60 * 60 / pixel_scale

        #make sure args.num_sections is a multiple of 18000 (pixels)
        image_width,image_height = int(total_width/args.num_sections),int(total_height/args.num_sections) 
        cmd='python {} --catalog-name {} --survey-name {} --image-width {} --image-height {} --output-name {}/{}_{}_{} --ra-center {} --dec-center {} --calculate-bias --cosmic-shear-g1 {} --cosmic-shear-g2 {} --verbose --no-stamps --no-agn --no-hsm --filter-band i'
        slac_cmd='bsub -M {} -W {}:00 -o "{}/output_{}_{}.txt" -r "{}"'

        for i,x in enumerate(np.linspace(endpoint1,endpoint2, args.num_sections)):
            for j,y in enumerate(np.linspace(endpoint1,endpoint2, args.num_sections)):

                curr_cmd = cmd.format(inputs['simulate_file'], inputs['one_sq_degree'],args.survey_name,image_width,image_height, inputs['project'], SECTION_NAME,i,j,x,y,args.cosmic_shear_g1,args.cosmic_shear_g2)

                curr_slac_cmd = slac_cmd.format(args.max_memory,args.bjob_time,inputs['project'],i,j,curr_cmd)

                logger.info(f"Running the slac cmd: {curr_slac_cmd}")

                os.system(curr_slac_cmd)
                

    elif args.simulate_single:
        raise NotImplementedError("For now this is not implemented.")
        cmd = './simulate.py --catalog-name {} --survey-name {} --image-width {} --image-height {} --output-name {} --calculate-bias --cosmic-shear-g1 {} --cosmic-shear-g2 {} --ra-center {} --dec-center {} --verbose --no-stamps --no-agn --no-hsm'.format(inputs['one_sq_degree'],args.survey_name,args.single_image_width,args.single_image_height,inputs['single_section'],args.cosmic_shear_g1,args.cosmic_shear_g2,args.single_image_ra_center,args.single_image_dec_center)
        slac_cmd = 'bsub -M {} -W {}:00 -o "{}/output-{}.txt" -r "{}"'.format(args.max_memory,args.bjob_time,inputs['project'],args.section_name,cmd)
        os.system(slac_cmd)

    ######################################################################################################################################################################################################################################
    #trim extra HDUs (containing individual galaxy partials, etc.) to reduce file size. 
    #this is obsolete if we just use the --no-stamps argument, and also better to use since never occupy too much space.


    # def process(file_name): 
    #     logger.info(f"Delete the third HDU and on from {file_name}")

    #     hdus = fits.open(file_name)
    #     del hdus[2:]
    #     subprocess.call('rm {}'.format(file_name),shell=True)  #delete old file 

    #     logger.info(f"Creating {file_name} w/out the HDUs previously deleted...")

    #     hdus.writeto(file_name)
    #     hdus.close()


    # if args.process_all: 
    #     for i in range(args.num_sections):
    #         for j in range(args.num_sections):
    #             file_name = '{}/{}_{}_{}.fits'.format(args.project,SECTION_NAME,i,j)
    #             process(file_name)


    # if args.process_single: 
    #     process(inputs['single_section'])


    ######################################################################################################################################################################################################################################
    #add noise to the image before extraction. 
    def add_noise(noisefile_name,file_name,noise_seed): 

        #take any noise_seed and add noise to the image generated
        logger.debug(f"Adding noise to {file_name} with noise seed: {noise_seed}")
        
        reader = descwl.output.Reader(file_name)
        results = reader.results
        results.add_noise(noise_seed=noise_seed)
        f = fits.PrimaryHDU(results.survey.image.array)

        logger.info(f"Writing new image with noise as .fits file to {noisefile_name}.")
        f.writeto(noisefile_name)

    if args.add_noise_all: 
        for i in range(args.num_sections):
            for j in range(args.num_sections):

                noisefile_name = '{}/{}_{}_{}.fits'.format(args.project,args.noise_name,i,j)
                file_name = '{}/{}_{}_{}.fits'.format(args.project,SECTION_NAME,i,j)
                add_noise(noisefile_name,file_name,args.noise_seed)

    if args.add_noise_single: 
        add_noise(inputs['noise_image'], inputs['single_section'], args.noise_seed)

    ###################################################################################################################
    #source extract and detect ambiguous blends 

    def extract(file_name,noisefile_name,outputfile_name,finalfits_name,total_height=None,total_width=None,x=None,y=None):

        logger.info(f"Will source extract galaxies from noise file {noisefile_name}")

        #run sextractor on noise image.
        cmd = 'sex {} -c {} -CATALOG_NAME {} -PARAMETERS_NAME {} -FILTER_NAME {} -STARNNW_NAME {}'.format(noisefile_name,inputs['config_file'],outputfile_name,inputs['param_file'],inputs['filter_file'],inputs['starnnw_file'])

        logger.info(f"With cmd: {cmd}")

        os.system(cmd)

        logger.success(f"Successfully source extracted {noisefile_name}!")

        #read noise image to figure out image bounds. 
        fits_section = fitsio.FITS(noisefile_name)
        stamp = fits_section[0].read()
        image_width = stamp.shape[0]
        image_height = stamp.shape[1]

        #read results obtained (table obtained either from combinining or from single.)
        cat = descwl.output.Reader(file_name).results
        table = cat.table
        detected,matched,indices,distance = cat.match_sextractor(outputfile_name)
        logger.success(f"Successfully matched catalogue with source extractor from sextract output: {outputfile_name}")

        #convert to arcsecs and relative to this image's center (not to absolute)
        detected['X_IMAGE'] = (detected['X_IMAGE'] - 0.5*image_width - 0.5)*pixel_scale
        detected['Y_IMAGE'] = (detected['Y_IMAGE'] - 0.5*image_height - 0.5)*pixel_scale

        #adjust to absolute image center if necessary, both for measured centers and catalogue centers. 
        if x!=None and y!=None and total_height!=None and total_width!=None: 
            detected['X_IMAGE']+=x*(total_width*pixel_scale)
            detected['Y_IMAGE']+=y*(total_height*pixel_scale)

            table['dx']+=x*(total_width*pixel_scale)
            table['dy']+=y*(total_height*pixel_scale)

        #convert second moments arcsecs, do not have to adjust because we only just it for sigma calculation. 
        detected['X2_IMAGE']*=pixel_scale**2 
        detected['Y2_IMAGE']*=pixel_scale**2 
        detected['XY_IMAGE']*=pixel_scale**2 

        # calculate size from moments X2_IMAGE,Y2_IMAGE,XY_IMAGE -> remember in pixel**2 so have to convert to arcsecs. 
        sigmas = []
        for x2,y2,xy in zip(detected['X2_IMAGE'],detected['Y2_IMAGE'],detected['XY_IMAGE']):
            second_moments = np.array([[x2,xy],[xy,y2]])
            sigma = np.linalg.det(second_moments)**(+1./4) #should be a PLUS. 
            sigmas.append(sigma)

        SIGMA = astroTable.Column(name='SIGMA',data=sigmas)
        detected.add_column(SIGMA)

        #find the indices of the ambiguous blends. 
        logger.info("Finding indices/ids that are ambiguosly blended")
        ambg_blends = detected_ambiguous_blends(table, indices, detected)
        ambg_blends_indices = set(list(ambg_blends))
        ambg_blends_ids = list(table[ambg_blends_indices]['db_id'])
        logger.success("All indices have been found")


        #add columns to table of undetected and ambiguosly blended 
        logger.info(f"Adding column to original table and writing it to {finalfits_name}")

        ambigous_blend_column = []
        for i,gal_row in enumerate(table):
            if i in ambg_blends_indices:
                ambigous_blend_column.append(True)
            else: 
                ambigous_blend_column.append(False)
        column = astroTable.Column(name='ambig_blend',data=ambigous_blend_column)
        table.add_column(column)

        logger.debug(f"Number of galaxies in table from file {finalfits_name} is: {len(table)}")

        table.write(finalfits_name)


    if args.extract_all: 
        total_height = 1. * 60 * 60 / pixel_scale #in pixels
        total_width = 1. * 60 * 60 / pixel_scale
        endpoint2 = (1.-1./args.num_sections)/2
        endpoint1 = -endpoint2

        logger.debug(f"Initializing extraction with total height {total_height} and total width {total_width} and endpoints.")

        for i,x in enumerate(np.linspace(endpoint1,endpoint2, args.num_sections)):
            for j,y in enumerate(np.linspace(endpoint1,endpoint2, args.num_sections)):
                file_name = '{}/{}_{}_{}.fits'.format(args.project,SECTION_NAME,i,j)
                noisefile_name = '{}/{}_{}_{}.fits'.format(args.project,args.noise_name,i,j)
                outputfile_name = '{}/{}_{}_{}.cat'.format(args.project,args.outcat_name,i,j)
                finalfits_name =  '{}/{}_{}_{}.fits'.format(args.project,args.final_name,i,j)
                extract(file_name,noisefile_name, outputfile_name, finalfits_name,total_height,total_width,x,y)

    if args.extract_single: 
        extract(inputs['single_section'],inputs['noise_image'], inputs['output_detected'], inputs['final_fits'])


    ##########################################################################################################################################################################################
    #combine 16 regions into image and table. 
    if args.combine:

        #############
        #constants used: 
        total_height = 1. * 60 * 60 / pixel_scale #pixels 
        total_width = 1. * 60 * 60 / pixel_scale
        #############

        tables = []
        #also have to combine the full galaxy postage stamp. 
        stamps,past_i = None,None
        for i in range(args.num_sections):
            for j in range(args.num_sections):
                finalfits_name =  '{}/{}_{}_{}.fits'.format(args.project,args.final_name,i,j)

                logger.info(f"Reading one of final fits before combining with name {finalfits_name}")

                table = astroTable.Table.read(finalfits_name)                
                tables.append(table)


        #combine tables list into final Table 
        logger.info("Vstacking the tables from each of the final fits files into one big table...")

        from astropy.table import vstack 
        Table = vstack(tables)
        Table.write(inputs['final_fits'])

        logger.success(f"Successfully stacked tables!, wrote them to {inputs['final_fits']}")
        logger.debug(f"Number of galaxies after combining all tables is {len(Table)}")

        #have to adjust to a correct header. 
        logger.debug(f"adjusting the headers in a weird way of the final fit file. Total width is {total_width} and total height {total_height}")

        f = fits.open(inputs['final_fits'])

        logger.info(f"Reading header from {inputs['sample_fits']}")

        f_sample = fits.open(inputs['sample_fits'])  #sample section of the job_number. 
        f[0].header = f_sample[0].header
        f[0].header['E_HEIGHT'] = total_height
        f[0].header['GE_WIDTH'] = total_width

        #delete older one so no problems at overwriting. 
        subprocess.call('rm {0}'.format(inputs['final_fits']), shell=True) 

        logger.info(f"Writing final results to {inputs['final_fits']}")

        f.writeto(inputs['final_fits'])

        logger.success("All done writing last file.")

if __name__=='__main__':
    main()




#read noise image. 
# fits_section = fitsio.FITS(noisefile_name)
# stamp = fits_section[0].read()
# generator = galsim.random.BaseDeviate(seed = noise_seed)
# noise = galsim.PoissonNoise(rng = generator, sky_level = results.survey.mean_sky_level)
# stamp_galsim = galsim.Image(array=stamp,wcs=galsim.PixelScale(pixel_scale),bounds=galsim.BoundsI(xmin=0, xmax=stamp.shape[0]-1, ymin=0, ymax=stamp.shape[1]-1))
# stamp_galsim.addNoise(noise)