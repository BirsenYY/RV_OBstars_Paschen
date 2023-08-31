import numpy as np
from astropy.io import fits
from specutils.spectra import Spectrum1D, SpectralRegion
from astropy import units as u
from specutils.fitting import fit_continuum
import matplotlib.pyplot as mplot

MAX_WV = 8800 #maximum wavelength limit when reading from .fits file 
MIN_WV = 8400 #minimum wavelength limit when reading from .fits file

#Excluded regions are used in continuum normalisation.
EXCLUDED_REGIONS = [ SpectralRegion((8400)*u.angstrom, (8420.1)*u.angstrom),
                     SpectralRegion((8428.1)*u.angstrom, (8480.9)*u.angstrom),
                     SpectralRegion((8486.5)*u.angstrom, (8520.8)*u.angstrom),
                     SpectralRegion((8527.2)*u.angstrom, (8626.3)*u.angstrom),
                     SpectralRegion((8628.7)*u.angstrom, (8701.4)*u.angstrom),
                     SpectralRegion((8709.4)*u.angstrom, (8798.9)*u.angstrom),
                     SpectralRegion((8806.9)*u.angstrom, (8800.0)*u.angstrom)]

'''Reads the fits files (spectra) and loads the content in a 2D array'''
def read_fits_file(fits_file_name):
    hdu = fits.open('fits_files/' + fits_file_name)
    data = hdu[1].data
    wavelength = data['Wavelength']
    flux = data['Count']
    spectrum_data = np.empty((len(wavelength), 2))
    spectrum_data[:, 0] = wavelength
    spectrum_data[:, 1] = flux
    spectrum_data = spectrum_data[~np.isnan(spectrum_data).any(axis=1)]
    spectrum_data = spectrum_data[np.logical_and(
        spectrum_data[:, 0] < MAX_WV, spectrum_data[:, 0] > MIN_WV), :]
    return spectrum_data

'''Reads text files and loads the content to an array'''
def file_to_array_txt(file_name):
    lines = np.loadtxt(file_name, str, unpack=False, delimiter=" ")
    spectrum_data = np.empty(lines.shape, float)
    x, y = spectrum_data.shape

    for i in range(0, x):
        spectrum_data[i][0] = float(lines[i][0])
        spectrum_data[i][1] = float(lines[i][1])
    return spectrum_data


'''Continuum normalze to zero, removal of cosmic hits'''
def preprocess(spectrum_data):
    spectrum = Spectrum1D(flux=spectrum_data[:, 1]*u.dimensionless_unscaled, 
               spectral_axis=spectrum_data[:, 0]*u.angstrom)
    continuum_fitted_spectrum = fit_continuum(spectrum, 
                                exclude_regions=EXCLUDED_REGIONS) 
    continuum_flux = continuum_fitted_spectrum(spectrum_data[:, 0]*u.angstrom)
    '''Run the following line to plot the continuum flux over the spectrum.'''
   # plot_continuum(spectrum_data, continuum_flux):
    spectrum_data[:, 1] = (spectrum_data[:, 1]/continuum_flux)
    spectrum_data[:, 1] -= 1
    spectrum_data[:, 1] = fixer(spectrum_data[:, 1], 5) 
    return spectrum_data

'''Calculated modified z scores of fluxes.'''
def modified_z_score(intensity):
   
    med = np.median(intensity)
    med_int = np.median([np.abs(intensity - med)])

    modified_z_scores = (0.6745 * (intensity - med) / med_int)
    return modified_z_scores

'''Identify the cosmic hits, then replace the cosmic hits with average
of nearby fluxes after ident.'''
def fixer(y, m):
    threshold = 4  # binarization threshold.
    spikes = abs(np.array(modified_z_score(np.diff(y)))) > threshold

    y_out = y.copy()  # So we don’t overwrite y
    for i in np.arange(len(spikes)):
        if spikes[i] != 0:  # If we have an spike in position i
            if i-m < 0 or i+1+m > len(spikes):
                pass
            else:
                # we select 2 m + 1 points around our spike
                w = np.arange(i-m, i+1+m)
                # From such interval, we choose the ones which are not spikes
                w2 = w[spikes[w] == 0]
                y_out[i] = np.mean(y[w2])  # and we average the value

    return y_out

'''Use to plot continuum flux superposed the spectrum.'''
def plot_continuum(spectrum_data, continuum_flux):
    mplot.plot(spectrum_data[:, 0], spectrum_data[:, 1], color='gray')
    mplot.plot(spectrum_data[:, 0], continuum_flux, color='b', linewidth=20)
    mplot.xlabel('Wavelength ($\mathbf{\AA}$)', fontweight='bold')
    mplot.ylabel('Count', fontweight='bold')
    mplot.show()


