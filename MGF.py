
import common_methods as cm
import numpy as np
import matplotlib.pyplot as mplot
from astropy.modeling import fitting, models

#The following parameters are half of 'Gaussian window'. They are used to 
#determine the wavelength limits of truncated spetral lines. 
GAUSSIAN_HALF_WINDOW_PA12 = 7.5
GAUSSIAN_HALF_WINDOW_PA13 = 5.5
GAUSSIAN_HALF_WINDOW_PA14 = 4.0
GAUSSIAN_HALF_WINDOW_PA15 = 3.0

PA12 = 8750.472 #Rest wavelength of Pa12
PA13 = 8665.018 #Rest wavelength of Pa13
PA14 = 8598.392 #Rest wavelength of Pa14
PA15 = 8545.383 #Rest wavelength of Pa12

rest_wv_dict= {
    "2": [8750.472, 8665.018],
    "3": [8750.472, 8665.018, 8598.392],
    "4": [8750.472, 8665.018, 8598.392, 8545.383]
    }


def calculate_RV(observed_WV, rest_WV):
    return ((observed_WV - rest_WV)/rest_WV) * 299792.458

'''The following 4 routines connect the centres of Pa13, Pa14 and P15 to 
the centre of Pa12 through Doppler shift relation.'''
def tie_Pa13_to_P12_centre(model):
    return model.mean_0 * PA13/ PA12

def tie_Pa14_to_P12_centre(model):
    return model.mean_0 * PA14/ PA12

def tie_Pa15_to_P12_centre(model):
    return model.mean_0 * PA15/ PA12

def tie_to_P12_stddev(model):
    return model.stddev_0

'''This routine can be used if wanted to display the Gaussian fits.'''
def plot_Gaussian_fits(spectrum_data, fitted_lines):
    mplot.plot(spectrum_data[:,0], spectrum_data[:,1], color = 'grey')
    mplot.plot(spectrum_data[:,0], fitted_lines, color="r", 
          label="Fitted Model", linewidth = 10) 
    mplot.xlabel('Wavelength  ($\AA$)', fontweight='bold', fontsize = 100)
    mplot.ylabel('Normalized Flux', fontweight='bold', fontsize = 100)
    mplot.xticks(fontsize = 70)
    mplot.yticks(fontsize = 70)
    mplot.show()
 
'''Multiple Gaussian fitting is carried out in this routine.'''
def multiple_Gaussian_fit(spectrum_data, rest_wvs, line_count):
    P12_data = spectrum_data[np.logical_and(
        spectrum_data[:, 0] < rest_wvs[0] + GAUSSIAN_HALF_WINDOW_PA12, 
        spectrum_data[:, 0] > rest_wvs[0] - GAUSSIAN_HALF_WINDOW_PA12), :]
   
    P13_data = spectrum_data[np.logical_and(
        spectrum_data[:, 0] < rest_wvs[1] + GAUSSIAN_HALF_WINDOW_PA13, 
        spectrum_data[:, 0] > rest_wvs[1] - GAUSSIAN_HALF_WINDOW_PA13), :]
    
    truncated_spectrum  = np.concatenate((P13_data, P12_data))
    
    P12_Gaussian = models.Gaussian1D(amplitude=-0.4, mean=rest_wvs[0],
                          stddev=5)
    P12_Gaussian.amplitude.max = 0.0
       
    P13_Gaussian = models.Gaussian1D(amplitude=-0.3, mean=rest_wvs[1], 
                          stddev = 5)
    P13_Gaussian.amplitude.max = 0.0
    P13_Gaussian.stddev.tied = tie_to_P12_stddev
    
    model = P12_Gaussian + P13_Gaussian
    P13_Gaussian.mean.tied = tie_Pa13_to_P12_centre
    
    if line_count == 3 or line_count == 4:
       P14_data = spectrum_data[np.logical_and(
            spectrum_data[:, 0] < rest_wvs[2] + GAUSSIAN_HALF_WINDOW_PA14,
            spectrum_data[:, 0] > rest_wvs[2] - GAUSSIAN_HALF_WINDOW_PA14), :]
       truncated_spectrum = np.concatenate((P14_data, truncated_spectrum))
      
       P14_Gaussian = models.Gaussian1D(amplitude=-0.2, mean=rest_wvs[2],
                             stddev=5)
   
       P14_Gaussian.amplitude.max= 0.0
       P14_Gaussian.stddev.tied = tie_to_P12_stddev
       
       model = model + P14_Gaussian 
       P14_Gaussian.mean.tied = tie_Pa14_to_P12_centre
       
    if line_count == 4:  
       P15_data = spectrum_data[np.logical_and(
            spectrum_data[:, 0] < rest_wvs[3] + GAUSSIAN_HALF_WINDOW_PA15, 
            spectrum_data[:, 0] > rest_wvs[3] - GAUSSIAN_HALF_WINDOW_PA15), :]
       truncated_spectrum = np.concatenate((P15_data, truncated_spectrum))
       P15_Gaussian = models.Gaussian1D(amplitude=-0.1, mean=rest_wvs[3], 
                             stddev = 5)
       P15_Gaussian.amplitude.max = 0.0
       P15_Gaussian.stddev.tied = tie_to_P12_stddev
       model = model + P15_Gaussian
       P15_Gaussian.mean.tied = tie_Pa15_to_P12_centre
    
    spectrum_data = spectrum_data[np.logical_and(
        spectrum_data[:, 0] < rest_wvs[0] + 50, 
        spectrum_data[:, 0] > rest_wvs[-1] - 50), :]
    fitter = fitting.LMLSQFitter(calc_uncertainties=True)
    fitted_model = fitter(model, truncated_spectrum[:,0], 
                   truncated_spectrum[:,1],   maxiter=3000)
    
    #Obtain errors (in wavelength)
    cov = fitter.fit_info['param_cov']
    errs = np.sqrt(np.diag(cov))

    '''The line below can be run to display the Gaussians with the spectrum.'''
    
    #plot_Gaussian_fits(spectrum_data, fitted_model(spectrum_data[:,0]))
    
    #Turn P12 centre from wavelength into RV
    rv= calculate_RV(fitted_model.mean_0, PA12) 
    
    #Convert the error from wavelength into radial velocity
    err = calculate_RV(errs[1] + PA12, PA12)
    return rv, err
    
'''Main body of the program starts here'''

file_names = np.loadtxt('stars_159.txt', str, unpack=False)

for line_count in range(2,5):#MGF2, MGF3, MGF2 implemented

    rv_file= open('MGF' + str(line_count)+'_RVs.txt', "w")
    rv_err_file = open('RV_err_' + str(line_count)+'RVs_err.txt', "w")
  
    rest_wvs = rest_wv_dict[str(line_count)]
   
    for j in range(0,len(file_names)):
      
        spectrum_data = cm.read_fits_file(file_names[j])
        cm.preprocess(spectrum_data)
  
        rv, err = multiple_Gaussian_fit(spectrum_data, rest_wvs, line_count)
             
        rv_file.write(str(np.round(rv, 4))+"\n")
        rv_err_file.write(str(np.round(err, 4))+"\n")

   
    rv_err_file.close()
    rv_file.close()




