import numpy as np 
import os
import sys 
import argparse
import descwl 
import subprocess
import fitsio
import astropy 
import astropy.io.fits as fits
import astropy.table as astroTable 

#algorithm for ambiguous blends - returns indices of ambiguously blended objects in terms of the table of full catalogue. 
#ambiguously blended means that at least one non-detected true object is less than a unit of effective distance away from an ambiguous object. 
def detected_ambiguous_blends(table,matched_indices,detected):
    """
    table: table containing entries of all galaxies of the catalog. 
    matched_indices: indices of galaxies in table that were detected by SExtractor (primary  matched)
    detected: Table of detected objects by SExtractor which also contains their information as measured by SExtractor. 
    """
    ambiguous_blends_indices = []
    for index,gal_row in enumerate(table):
        if gal_row['match'] == -1: #loop through true not primarily matched objects. 
            
            #calculate of effective distances of true non-primary match object with all primarily matched objects. 
            effective_distances = list(np.sqrt((detected['X_IMAGE']-gal_row['dx'])**2 + (detected['Y_IMAGE']-gal_row['dy'])**2)/(detected['SIGMA'] + gal_row['psf_sigm']))
            
            #mark all objects with effective distance <1. as ambiguosly blended. 
            marked_indices = [matched_indices[i] for i,distance in enumerate(effective_distances) if distance < 1.]
            
            #if at least one index was marked as ambiguous, add both the primarily match object and the true non-primary object as ambiguous. 
            if len(marked_indices) > 0:
                ambiguous_blends_indices.append(index)
                ambiguous_blends_indices.extend(marked_indices)
    return set(ambiguous_blends_indices)


def main():
    parser = argparse.ArgumentParser(description=('Simulate 16 different regions from a square' 
                                                  'degree and analyze their combination in'
                                                  'SExtractor.'),
                                     formatter_class=(
                                     argparse.ArgumentDefaultsHelpFormatter))

    parser.add_argument('--simulate-all', action='store_true',
                        help=('Simulates the requested 16 regions with given job_number'))

    parser.add_argument('--add-noise-all', action='store_true',
                        help=('Add noise to all of the images (one for each section) of the project.'))

    parser.add_argument('--extract-all', action='store_true',
                        help=('Classifies into detected and ambiously blended for a one square degree.'))

    parser.add_argument('--combine', action='store_true',
                        help=('Combines 16 regions with given job_number'))

    parser.add_argument('--process-all', action='store_true',
                        help=('Prepare files needed to run pipeline in one square degree .'))


    parser.add_argument('--simulate-single', action='store_true',
                        help=('Simulate a single chip with width,height specified with single-image-height,etc.'))

    parser.add_argument('--process-single', action='store_true',
                        help=('Prepare files needed to run pipeline in a single chip.'))

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

    SECTION_NAME = 'section'

    ###################################################################
    #names to be used.
    inputs = dict(
    noise_image = '/Users/Ismael/aegis/WeakLensingDeblending/{}/{}.fits'.format(args.project,args.noise_name),
    output_detected = '/Users/Ismael/aegis/WeakLensingDeblending/{}/{}.cat'.format(args.project,args.outcat_name),
    final_fits = '/Users/Ismael/aegis/WeakLensingDeblending/{}/{}.fits'.format(args.project,args.final_name),
    project = '/Users/Ismael/aegis/WeakLensingDeblending/{}'.format(args.project),
    single_section = '/Users/Ismael/aegis/WeakLensingDeblending/{}/{}.fits'.format(args.project,args.section_name),
    config_file = '/Users/Ismael/aegis/data/sextractor_runs/default.sex',
    param_file = '/Users/Ismael/aegis/data/sextractor_runs/default.param',
    filter_file = '/Users/Ismael/aegis/data/sextractor_runs/default.conv', 
    starnnw_file = '/Users/Ismael/aegis/data/sextractor_runs/default.nnw',
    WLD = '/Users/Ismael/aegis/WeakLensingDeblending/',
    sample_fits = '/Users/Ismael/aegis/data/section001.fits',
    one_sq_degree = '/Users/Ismael/aegis/data/OneDegSq.fits',
    )

    #change names to slac names to be used inside slac. 
    inputs_slac = dict()
    for f in inputs: 
        l = inputs[f].split("/")
        index_aegis = l.index("aegis")
        str_slac = "/".join(l[index_aegis+1:])
        slac_file = '{0}{1}'.format('/nfs/slac/g/ki/ki19/deuce/AEGIS/ismael/',str_slac)
        inputs_slac[f] = slac_file

    #create project directory if does not exist yet
    if not os.path.exists(inputs_slac['project']):
        os.makedirs(inputs_slac['project'])

    os.chdir(inputs_slac['WLD'])
    ##########################################################################################################################################################################################
    #simulate the 16 regions. 

    if args.simulate_all:
        #constants used for the endpoints. 
        endpoint2 = (1.-1./args.num_sections)/2
        endpoint1 = -endpoint2
        total_height = 1. * 60 * 60 / pixel_scale #in pixels
        total_width = 1. * 60 * 60 / pixel_scale

        image_width,image_height = int(total_width/args.num_sections),int(total_height/args.num_sections) #preferentially args.num_sections is a multiple of 18000 (pixels)

        for i,x in enumerate(np.linspace(endpoint1,endpoint2, args.num_sections)):
            for j,y in enumerate(np.linspace(endpoint1,endpoint2, args.num_sections)):
                cmd = './simulate.py --catalog-name {} --survey-name {} --image-width {} --image-height {} --output-name {}/{}{}{} --ra-center {} --dec-center {} --calculate-bias --cosmic-shear-g1 {} --cosmic-shear-g2 {} --verbose'.format(inputs_slac['one_sq_degree'],args.survey_name,image_width,image_height,args.project,SECTION_NAME,i,j,x,y,args.cosmic_shear_g1,args.cosmic_shear_g2)
                slac_cmd = 'bsub -M {} -W {}:00 -o "{}/output{}{}.txt" -r "{}"'.format(args.max_memory,args.bjob_time,inputs_slac['project'],i,j,cmd)
                os.system(slac_cmd)

    elif args.simulate_single:
        cmd = './simulate.py --catalog-name {} --survey-name {} --image-width {} --image-height {} --output-name {} --calculate-bias --cosmic-shear-g1 {} --cosmic-shear-g2 {} --ra-center {} --dec-center {} --verbose'.format(inputs_slac['one_sq_degree'],args.survey_name,args.single_image_width,args.single_image_height,inputs_slac['single_section'],args.cosmic_shear_g1,args.cosmic_shear_g2,args.single_image_ra_center,args.single_image_dec_center)
        slac_cmd = 'bsub -M {} -W {}:00 -o "{}/output-{}.txt" -r "{}"'.format(args.max_memory,args.bjob_time,inputs_slac['project'],args.section_name,cmd)
        os.system(slac_cmd)

    ##########################################################################################################################################################################################
    #trim extra HDUs to reduce file size. 
    def process(file_name): 
        hdus = astropy.io.fits.open(file_name)
        del hdus[2:]
        
        #delete old file 
        subprocess.call('rm {}'.format(file_name),shell=True)
        hdus.writeto(file_name)
        hdus.close()


    if args.process_all: 
        for i in range(args.num_sections):
            for j in range(args.num_sections):
                file_name = '{}/{}{}{}.fits'.format(args.project,SECTION_NAME,i,j)
                process(file_name)


    if args.process_single: 
        process(inputs_slac['single_section'])


    #######################################################################################################################################################################################
    #add noise to the image before extraction. 
    def add_noise(noisefile_name,file_name,noise_seed): 
        #read noise image. 
        # fits_section = fitsio.FITS(noisefile_name)
        # stamp = fits_section[0].read()
        # generator = galsim.random.BaseDeviate(seed = noise_seed)
        # noise = galsim.PoissonNoise(rng = generator, sky_level = results.survey.mean_sky_level)
        # stamp_galsim = galsim.Image(array=stamp,wcs=galsim.PixelScale(pixel_scale),bounds=galsim.BoundsI(xmin=0, xmax=stamp.shape[0]-1, ymin=0, ymax=stamp.shape[1]-1))
        # stamp_galsim.addNoise(noise)

        #take any noise_seed and add noise to the image generated
        reader = descwl.output.Reader(file_name)
        results = reader.results
        results.add_noise(noise_seed=noise_seed)
        f = fits.PrimaryHDU(results.survey.image.array)
        f.writeto(noisefile_name)


    if args.add_noise_all: 
        for i in range(args.num_sections):
            for j in range(args.num_sections):

                noisefile_name = '{}/{}{}{}.fits'.format(args.project,args.noise_name,i,j)
                file_name = '{}/{}{}{}.fits'.format(args.project,SECTION_NAME,i,j)
                add_noise(noisefile_name,file_name,args.noise_seed)

    if args.add_noise_single: 
        add_noise(inputs_slac['noise_image'], inputs_slac['single_section'], args.noise_seed)

    ##########################################################################################################################################################################################
    #source extract and detect ambiguous blends 

    def extract(file_name,noisefile_name,outputfile_name,finalfits_name,total_height=None,total_width=None,x=None,y=None):

        #run sextractor on noise image.
        cmd = 'sex {} -c {} -CATALOG_NAME {} -PARAMETERS_NAME {} -FILTER_NAME {} -STARNNW_NAME {}'.format(noisefile_name,inputs_slac['config_file'],outputfile_name,inputs_slac['param_file'],inputs_slac['filter_file'],inputs_slac['starnnw_file'])
        print(cmd)
        os.system(cmd)

        #read noise image to figure out image bounds. 
        fits_section = fitsio.FITS(noisefile_name)
        stamp = fits_section[0].read()
        image_width = stamp.shape[0]
        image_height = stamp.shape[1]

        #read results obtained (table obtained either from combinining or from single.)
        cat = descwl.output.Reader(file_name).results
        table = cat.table
        detected,matched,indices,distance = cat.match_sextractor(outputfile_name)

        #convert to arcsecs and relative to this image's center (not to absolute)
        detected['X_IMAGE'] = (detected['X_IMAGE'] - 0.5*image_width - 0.5)*pixel_scale
        detected['Y_IMAGE'] = (detected['Y_IMAGE'] - 0.5*image_height - 0.5)*pixel_scale

        #adjust to absolute image center if necessary, both for measured centers and catalogue centers. 
        if x!=None and y!=None and total_height!=None and total_width!=None: 
            detected['X_IMAGE']+=x*(total_width*pixel_scale)
            detected['Y_IMAGE']+=y*(total_height*pixel_scale)

            table['dx']+=x*total_width*pixel_scale 
            table['dy']+=y*total_height*pixel_scale

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
        ambg_blends = detected_ambiguous_blends(table, indices, detected)
        ambg_blends_indices = list(ambg_blends)
        ambg_blends_ids = list(table[ambg_blends_indices]['db_id'])

        #add columns to table of undetected and ambiguosly blended 
        ambigous_blend_column = []
        for i,gal_row in enumerate(table):
            if i in ambg_blends_indices:
                ambigous_blend_column.append(True)
            else: 
                ambigous_blend_column.append(False)
        column = astroTable.Column(name='ambig_blend',data=ambigous_blend_column)
        table.add_column(column)

        table.write(finalfits_name)

    if args.extract_all: 

        total_height = 1. * 60 * 60 / pixel_scale #in pixels
        total_width = 1. * 60 * 60 / pixel_scale
        endpoint2 = (1.-1./args.num_sections)/2
        endpoint1 = -endpoint2

        for i,x in enumerate(np.linspace(endpoint1,endpoint2, args.num_sections)):
            for j,y in enumerate(np.linspace(endpoint1,endpoint2, args.num_sections)):
                file_name = '{}/{}{}{}.fits'.format(args.project,SECTION_NAME,i,j)
                noisefile_name = '{}/{}{}{}.fits'.format(args.project,args.noise_name,i,j)
                outputfile_name = '{}/{}{}{}.cat'.format(args.project,args.outcat_name,i,j)
                finalfits_name =  '{}/{}{}{}.fits'.format(args.project,args.final_name,i,j)
                extract(file_name,noisefile_name, outputfile_name, finalfits_name,total_height,total_width,x,y)

    if args.extract_single: 
        extract(inputs_slac['single_section'],inputs_slac['noise_image'], inputs_slac['output_detected'], inputs_slac['final_fits'])


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
                finalfits_name =  '{}/{}{}{}.fits'.format(args.project,args.final_name,i,j)
                table = astroTable.Table.read(finalfits_name)                
                tables.append(table)


        #combine tables list into final Table 
        from astropy.table import vstack 
        Table = vstack(tables)
        Table.write(inputs_slac['final_fits'])

        #have to adjust to a correct header. 
        f = fits.open(inputs_slac['final_fits'])
        f_sample = fits.open(inputs_slac['sample_fits'])  #sample section of the job_number. 
        f[0].header = f_sample[0].header
        f[0].header['E_HEIGHT'] = total_height
        f[0].header['GE_WIDTH'] = total_width
        subprocess.call('rm {0}'.format(inputs_slac['final_fits']), shell=True) #delete older one so no problems at overwriting. 
        f.writeto(inputs_slac['final_fits'])

if __name__=='__main__':
    main()