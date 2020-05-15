#!/usr/bin/env python3
"""
This file is designed to run the fisher bias analysis on a one square degree simulation of WeakLensingDeblending.
It works by separating the one square degree into patches and doing the analysis in parallel.
For now it only works on the SLAC cluster.
"""

import argparse
import os
import subprocess

import astropy.io.fits as fits
import astropy.table
import fitsio
import numpy as np
from loguru import logger

from WeakLensingDeblending import descwl
from src import utils

SECTION_NAME = 'section'


# add noise to the image before extraction.
def add_noise(noisefile, filename, noise_seed):
    # take any noise_seed and add noise to the image generated
    logger.debug(f"Adding noise to {filename} with noise seed: {noise_seed}")

    reader = descwl.output.Reader(filename)
    results = reader.results
    results.add_noise(noise_seed=noise_seed)
    f = fits.PrimaryHDU(results.survey.image.array)

    logger.info(f"Writing new image with noise as .fits file to {noisefile}.")
    f.writeto(noisefile)


@logger.catch
def simulate_all(args, inputs, pixel_scale):
    # constants used for the endpoints.
    endpoint2 = (1. - 1. / args.num_sections) / 2  # in degrees.
    endpoint1 = -endpoint2
    total_height = 1. * 60 * 60 / pixel_scale  # in pixels
    total_width = 1. * 60 * 60 / pixel_scale

    assert total_width % args.num_sections == 0 and total_height % args.num_sections == 0

    image_width, image_height = int(total_width / args.num_sections), int(total_height / args.num_sections)
    cmd = 'python {} --catalog-name {} --survey-name {} --image-width {} --image-height {} ' \
          '--output-name {} --ra-center {} --dec-center {} --calculate-bias' \
          ' --cosmic-shear-g1 {} --cosmic-shear-g2 {} --verbose --no-stamps ' \
          '--no-agn --no-hsm --filter-band i --variations-g {} --equilibrate'

    if args.memory_trace:
        cmd += ' --memory-trace'

    slac_cmd = 'bsub -M {} -W {} -o "{}/output_{}_{}.txt" -r "{}"'

    for i, x in enumerate(np.linspace(endpoint1, endpoint2, args.num_sections)):
        for j, y in enumerate(np.linspace(endpoint1, endpoint2, args.num_sections)):

            output_name = "{}/{}_{}_{}".format(inputs['project'], SECTION_NAME, i, j)
            output_file = output_name + '.fits'

            if os.path.exists(output_file):
                if os.path.getsize(output_file) > 0:
                    continue
                else:
                    if os.path.exists(f"{inputs['project']}/output_{i}_{j}.txt"):
                        subprocess.run(f"rm {inputs['project']}/output_{i}_{j}.txt", shell=True)

            curr_cmd = cmd.format(inputs['simulate_file'], inputs['one_sq_degree'], args.survey_name, image_width,
                                  image_height, output_name, x, y, args.cosmic_shear_g1, args.cosmic_shear_g2,
                                  args.variations_g)

            curr_slac_cmd = slac_cmd.format(args.max_memory, args.bjob_time, inputs['project'], i, j, curr_cmd)

            logger.info(f"Running the slac cmd: \n {curr_slac_cmd}")

            if not args.testing:
                subprocess.run(curr_slac_cmd, shell=True)


@logger.catch
def simulate_single(args, inputs):
    assert args.num_sections == 1, "Remember that --simulate-single creates a single section."

    cmd = f"python {inputs['simulate_file']} --catalog-name {inputs['one_sq_degree']} " \
          f"--survey-name {args.survey_name} --image-width {args.single_image_width} " \
          f"--image-height {args.single_image_height} --output-name {inputs['single_section']} " \
          f"--ra-center 0 --dec-center 0 --calculate-bias --cosmic-shear-g1 {args.cosmic_shear_g1} " \
          f"--cosmic-shear-g2 {args.cosmic_shear_g2} --verbose --no-stamps --no-agn --no-hsm " \
          f"--filter-band i --variations-g {args.variations_g} "

    output_file = f"{inputs['project']}/output_sim.txt"

    slac_cmd = f'bsub -M {args.max_memory} -W {args.bjob_time} -o {output_file} -r "{cmd}"'
    logger.info(f"Running the slac cmd: {slac_cmd}")

    subprocess.run(slac_cmd, shell=True)


@logger.catch
def extract(inputs, pixel_scale, filename, noisefile, outputfile, finalfits, num_sections, height, width, x,
            y):
    logger.info(f"Will source extract galaxies from noise file {noisefile}")

    # run sextractor on noise image.
    cmd = '/nfs/slac/g/ki/ki19/deuce/AEGIS/ismael/WLD/params/sextractor-2.25.0/src/sex {} -c {} -CATALOG_NAME {} ' \
          '-PARAMETERS_NAME {} -FILTER_NAME {} -STARNNW_NAME {}'.format(
            noisefile, inputs['config_file'], outputfile, inputs['param_file'], inputs['filter_file'],
            inputs['starnnw_file'])

    logger.info(f"With cmd: {cmd}")

    subprocess.run(cmd, shell=True)

    logger.success(f"Successfully source extracted {noisefile}!")

    # read noise image to figure out image bounds.
    fits_section = fitsio.FITS(noisefile)
    stamp = fits_section[0].read()
    img_width = stamp.shape[0]  # pixels
    img_height = stamp.shape[1]

    assert img_width == width / num_sections and img_height == height / num_sections, \
        "Something is wrong with the heights specified."

    logger.info(f"The image width and height from the stamp are {img_width} and {img_height} respectively.")

    # read results obtained (table obtained either from combining or from single.)
    cat = descwl.output.Reader(filename).results
    table = cat.table
    detected, matched, indices, distance = cat.match_sextractor(outputfile)
    logger.success(f"Successfully matched catalogue with source extractor from sextract output: {outputfile}")

    # convert to arcsecs and relative to this image's center (not to absolute, before w/ respect to corner.)
    detected['X_IMAGE'] = (detected['X_IMAGE'] - 0.5 * img_width - 0.5) * pixel_scale
    detected['Y_IMAGE'] = (detected['Y_IMAGE'] - 0.5 * img_height - 0.5) * pixel_scale

    # adjust to absolute image center if necessary, both for measured centers and catalogue centers.
    detected['X_IMAGE'] += x * (width * pixel_scale)  # need to use arcsecs
    detected['Y_IMAGE'] += y * (height * pixel_scale)

    table['dx'] += x * (width * pixel_scale)
    table['dy'] += y * (height * pixel_scale)

    # also adjust the xmin,xmax, ymin, ymax bounding box edges, because why not?
    # remember these are in pixels; xmin, xmax relative to left edge; ymin,ymax relative to right edge.
    table['xmin'] += int(x * width + width / 2 - img_width / 2)
    table['xmax'] += int(x * width + width / 2 - img_width / 2)
    table['ymin'] += int(x * height + height / 2 - img_height / 2)
    table['ymax'] += int(x * height + height / 2 - img_height / 2)

    # convert second moments to arcsecs.
    # Not adjust relative to center, because only used for sigma calculation.
    detected['X2_IMAGE'] *= pixel_scale ** 2
    detected['Y2_IMAGE'] *= pixel_scale ** 2
    detected['XY_IMAGE'] *= pixel_scale ** 2

    # calculate size from moments X2_IMAGE,Y2_IMAGE,XY_IMAGE -> remember in pixel**2 so have to convert to arcsecs.
    sigmas = []
    for x2, y2, xy in zip(detected['X2_IMAGE'], detected['Y2_IMAGE'], detected['XY_IMAGE']):
        second_moments = np.array([[x2, xy], [xy, y2]])
        sigma = np.linalg.det(second_moments) ** (+1. / 4)  # should be a PLUS.
        sigmas.append(sigma)

    SIGMA = astropy.table.Column(name='SIGMA', data=sigmas)
    detected.add_column(SIGMA)

    # find the indices of the ambiguous blends.
    logger.info("Finding indices/ids that are ambiguously blended")
    ambg_blends = detected_ambiguous_blends(table, indices, detected)

    logger.success("All indices have been found")

    logger.info(f"Adding column to original table and writing it to {finalfits}")

    ambiguous_blend_column = []
    for i, gal_row in enumerate(table):
        if i in ambg_blends:
            ambiguous_blend_column.append(True)
        else:
            ambiguous_blend_column.append(False)
    column = astropy.table.Column(name='ambig_blend', data=ambiguous_blend_column)
    table.add_column(column)

    logger.debug(f"Number of galaxies in table from file {finalfits} is: {len(table)}")

    table.write(finalfits)


@logger.catch
def detected_ambiguous_blends(table, matched_indices, detected):
    """
    Algorithm for ambiguous blends - returns indices of ambiguously blended objects in terms of the table
    of full catalogue.

    Ambiguously blended means that at least one non-detected true object is less than a unit of
    effective distance away from an ambiguous object.

    table: table containing entries of all galaxies of the catalog.
    matched_indices: indices of galaxies in table that were detected by SExtractor (primary  matched)
    detected: Table of detected objects by SExtractor which also contains their information as measured
    by SExtractor.
    """
    ambiguous_blends_indices = []
    for index, gal_row in enumerate(table):
        if gal_row['match'] == -1:  # loop through true not primarily matched objects.

            # calculate of effective distances of true non-primary match object with all primarily matched objects.
            effective_distances = list(
                np.sqrt((detected['X_IMAGE'] - gal_row['dx']) ** 2 + (detected['Y_IMAGE'] - gal_row['dy']) ** 2) / (
                        detected['SIGMA'] + gal_row['psf_sigm']))

            # mark all objects with effective distance <1 as ambiguously blended.
            marked_indices = [matched_indices[i] for i, distance in enumerate(effective_distances) if distance < 1.]

            # if at least one index was marked as ambiguous, add both the primarily match object
            # and the true non-primary object as ambiguous.
            if len(marked_indices) > 0:
                ambiguous_blends_indices.append(index)
                ambiguous_blends_indices.extend(marked_indices)
    return set(ambiguous_blends_indices)


@logger.catch
def combine(args, pixel_scale, inputs):
    #############
    # constants used:
    total_height = 1. * 60 * 60 / pixel_scale  # pixels
    total_width = 1. * 60 * 60 / pixel_scale
    #############

    tables = []
    # also have to combine the full galaxy postage stamp.
    stamps, past_i = None, None
    for i in range(args.num_sections):
        for j in range(args.num_sections):
            finalfits_name = '{}/{}_{}_{}.fits'.format(inputs['project'], args.final_name, i, j)

            logger.info(f"Reading one of final fits before combining with name {finalfits_name}")

            t = astropy.table.Table.read(finalfits_name)
            tables.append(t)

    # combine tables list into final Table
    logger.info("Vertically stacking the tables from each of the final fits files into one big table...")

    from astropy.table import vstack
    Table = vstack(tables)
    Table.write(inputs['final_fits'])

    logger.success(f"Successfully stacked tables!, wrote them to {inputs['final_fits']}")
    logger.debug(f"Number of galaxies after combining all tables is {len(Table)}")
    logger.success("All done writing last file.")


@logger.catch
def main(args):
    # some constants used throughout

    if args.survey_name == 'LSST':
        pixel_scale = .2
    elif args.survey_name == 'HSC':
        pixel_scale = 0.17
    elif args.survey_name == 'DES':
        pixel_scale = .263
    else:
        raise ValueError

    logger.info(f"\n Using survey {args.survey_name} and pixel scale {pixel_scale}")

    ###################################################################

    logger.info(f"Remember by default we assume we are running in SLAC cluster.")

    inputs = dict(
        project=f'{utils.data_dir}/{args.project}',
        noise_image=f'{utils.data_dir}/{args.project}/{args.noise_name}.fits',
        final_fits=f'{utils.data_dir}/{args.project}/{args.final_name}.fits',
        single_section=f'{utils.data_dir}/{args.project}/{args.section_name}.fits',
        output_detected=f'{utils.data_dir}/{args.project}/{args.outcat_name}.cat',
        config_file=f'{utils.params_dir}/sextractor_runs/default.sex',
        param_file=f'{utils.params_dir}/sextractor_runs/default.param',
        filter_file=f'{utils.params_dir}/sextractor_runs/default.conv',
        starnnw_file=f'{utils.params_dir}/sextractor_runs/default.nnw',
        WLD=utils.root_dir,
        simulate_file=f'{utils.root_dir}/WeakLensingDeblending/simulate.py',
        one_sq_degree=f'{utils.params_dir}/OneDegSq.fits',
    )

    # create project directory if does not exist yet
    if not os.path.exists(inputs['project']):
        os.makedirs(inputs['project'])

    os.chdir(inputs['WLD'])  # convenience so we can run simulate...

    # simulate the regions.
    if args.simulate_all:
        simulate_all(args, inputs, pixel_scale)

    elif args.simulate_single:
        simulate_single(args, inputs, pixel_scale)

    if args.add_noise_all:
        for i in range(args.num_sections):
            for j in range(args.num_sections):
                noisefile_name = '{}/{}_{}_{}.fits'.format(inputs['project'], args.noise_name, i, j)
                file_name = '{}/{}_{}_{}.fits'.format(inputs['project'], SECTION_NAME, i, j)
                add_noise(noisefile_name, file_name, args.noise_seed)

    if args.add_noise_single:
        add_noise(inputs['noise_image'], inputs['single_section'], args.noise_seed)

    if args.extract_all:
        total_height = 1. * 60 * 60 / pixel_scale  # in pixels
        total_width = 1. * 60 * 60 / pixel_scale
        endpoint2 = (1. - 1. / args.num_sections) / 2
        endpoint1 = -endpoint2

        logger.debug(
            f"Initializing extraction with total height {total_height} and total width {total_width} and endpoints.")

        for i, x in enumerate(np.linspace(endpoint1, endpoint2, args.num_sections)):
            for j, y in enumerate(np.linspace(endpoint1, endpoint2, args.num_sections)):
                file_name = '{}/{}_{}_{}.fits'.format(inputs['project'], SECTION_NAME, i, j)
                noisefile_name = '{}/{}_{}_{}.fits'.format(inputs['project'], args.noise_name, i, j)
                outputfile_name = '{}/{}_{}_{}.cat'.format(inputs['project'], args.outcat_name, i, j)
                finalfits_name = '{}/{}_{}_{}.fits'.format(inputs['project'], args.final_name, i, j)
                extract(file_name, noisefile_name, outputfile_name, finalfits_name, args.num_sections, total_height,
                        total_width, x, y)

    if args.extract_single:
        raise NotImplementedError("not implemented yet...")
        extract(inputs['single_section'], inputs['noise_image'], inputs['output_detected'], inputs['final_fits'])

    # combine 16 regions into image and table.
    if args.combine:
        combine(args, pixel_scale, inputs)


if __name__ == '__main__':

    # setup logger.
    logger.add(f"{utils.logs_dir}/all-process.log", format="{time}-{level}: {message}", level="INFO", backtrace=True,
               rotation="7:00", enqueue=True)  # new file created every day at 7:00 am .

    # setup argparser.
    parser = argparse.ArgumentParser(description='Simulate different regions from a square degree and analyze their '
                                                 'combination in SExtractor.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments.
    parser.add_argument('--project', required=True,
                        type=str,
                        help='Project name, directory name for all your files.')

    parser.add_argument('--num-sections', required=True,
                        type=int,
                        help='Number of squares along one dimension to divide the one square degree.')

    # main commands that run multiple simulations and combine them.
    parser.add_argument('--simulate-all', action='store_true',
                        help='Simulates the requested regions with given job_number using multiple batch jobs.')

    parser.add_argument('--add-noise-all', action='store_true',
                        help='Add noise to all of the images (one for each section) of the project.')

    parser.add_argument('--extract-all', action='store_true',
                        help='Classifies into detected and ambiguously blended for a one square degree.')

    parser.add_argument('--combine', action='store_true',
                        help='Combines regions with given job_number')

    # commands that run a single simulation, used for testing.
    parser.add_argument('--simulate-single', action='store_true',
                        help='Simulate a single chip with width,height specified with single-image-height,etc.')

    parser.add_argument('--extract-single', action='store_true',
                        help='Classifies into detected and ambiguously blended for a single section.')

    parser.add_argument('--add-noise-single', action='store_true',
                        help=(
                            'Create a noise image for an image .fits file in current project. Have to pass in the '
                            'name of the image to add noise to.'))

    parser.add_argument('--single-image-width', default=None,
                        type=int,
                        help='Width of single image to be simulated (in pixels).')

    parser.add_argument('--single-image-height', default=None,
                        type=int,
                        help='Height of single image to be simulated (in pixels).')

    parser.add_argument('--single-image-ra-center', default=0.0,
                        type=float,
                        help='Ra center of the single image to be simulated')

    parser.add_argument('--single-image-dec-center', default=0.0,
                        type=float,
                        help='Dec center of the single image to be simulated')

    parser.add_argument('--section-name', default='section',
                        type=str,
                        help='specify name of the section image when it is a single one.')

    # arguments for specifying simulation.
    parser.add_argument('--cosmic-shear-g1', default=None,
                        type=float,
                        help='cosmic shear g1')

    parser.add_argument('--cosmic-shear-g2', default=None,
                        type=float,
                        help='cosmic shear g2')

    parser.add_argument('--variations-g', default=0.03,
                        type=float,
                        help='Step size for shear to use in simulation.')

    parser.add_argument('--noise-seed', default=None,
                        type=int,
                        help='noise seed to use when adding noise to image SExtracted.')

    parser.add_argument('--survey-name', required=True,
                        type=str, choices=['LSST', 'HSC', 'DES'],
                        help='Select survey to use.')

    # additional names and arguments for running remotely.
    parser.add_argument('--bjob-time', default='02:00',
                        type=str,
                        help='Time of jobs to be run in SLAC')

    parser.add_argument('--max-memory', default='4096MB',
                        type=str,
                        help='Max memory to be used when running a slac process.')

    parser.add_argument('--noise-name', default='noise_image',
                        type=str,
                        help='specify name of the noise image')

    parser.add_argument('--outcat-name', default='outcat',
                        type=str,
                        help='specify name of the output catalogue from sextracting.')

    parser.add_argument('--final-name', default='final_fits',
                        type=str,
                        help='specify name of the final fits')

    parser.add_argument('--memory-trace', action='store_true',
                        help="Whether to trace memory of individual simulate commands.")

    parser.add_argument('--testing', action='store_true')

    pargs = parser.parse_args()

    # make sure only arguments that make sense are selected.
    if (pargs.simulate_all or pargs.simulate_single) and (
            pargs.cosmic_shear_g1 is None or pargs.cosmic_shear_g2 is None):
        raise RuntimeError('Need to include the cosmic shear when simulating.')

    if (pargs.simulate_all or pargs.combine or pargs.add_noise_all or pargs.extract_all) and not pargs.num_sections:
        raise RuntimeError('Need to include number of sections when dealing with whole one square degree.')

    if (pargs.add_noise_single or pargs.add_noise_all) and pargs.noise_seed is None:
        raise RuntimeError('To add noise you need the noise seed!')

    if pargs.simulate_single and (pargs.single_image_height is None or pargs.single_image_width is None):
        raise RuntimeError('Need size of the image when processing a single section, default is 4096.')

    main()
