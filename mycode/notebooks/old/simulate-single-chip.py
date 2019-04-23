#to simulate the following files need whole weaklensing deblending package to run.
#need OneDegSq.fits, found in ftp://ftp.slac.stanford.edu/groups/desc/WL/

os.chdir('/Users/Ismael/code/lensing/WeakLensingDeblending/')

filename = ''
section_name = 'section-{}.fits'.format(filename)

noise_iamge = 'noise_image.fits'
table_file = 'table.fits'


cosmic_shear_g1 = 0. 
cosmic_shear_g2 = 0. 
job_number = 1
i,j = 0,0
image_width = 4500
image_height = 4500

cmd = './simulate.py --catalog-name OneDegSq.fits --survey-name LSST --image-width {} --image-height {} --output-name section-{}{}{} --calculate-bias --cosmic-shear-g1 {} --cosmic-shear-g2 {} --verbose'.format(image_width,image_height,i,j,job_number,cosmic_shear_g1,cosmic_shear_g2)
os.system(cmd)

noise_seed = 0
#get image
fits_section = fitsio.FITS(file_name)
stamp = fits_section[0].read()


reader = descwl.output.Reader(section_name)
results = reader.results
generator = galsim.random.BaseDeviate(seed = noise_seed)
noise = galsim.PoissonNoise(rng = generator, sky_level = results.survey.mean_sky_level)
full_stamp_galsim = galsim.Image(array=stamp,wcs=galsim.PixelScale(pixel_scale),bounds=galsim.BoundsI(xmin=0, xmax=total_width-1, ymin=0, ymax=total_height-1)
full_stamp_galsim.addNoise(noise)

                                 
f = fits.PrimaryHDU(full_stamp_galsim.array)
f.writeto(inputs_slac['noise_image'])