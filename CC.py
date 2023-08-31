import common_methods as cm
import numpy as np
from PyAstronomy.pyasl import crosscorrRV
import matplotlib.pyplot as mplot
import math


RV_MIN = -250
RV_MAX = 250
EDGE_SIZE = 25
STEP_SIZE = 8.4
PRB_FIT_HALF_WINDOW = 100
REST_WV_LIST = {
    "2": [8750.472, 8665.018],
    "3": [8750.472, 8665.018, 8598.392],
    "4": [8750.472, 8665.018, 8598.392, 8545.383]}


def display_parabolicfit(coeff, rvs, ccf, rv_ccf_trimmed):   
    fit_ccf = rv_ccf_trimmed[:,0]**2 * coeff[0] + \
        coeff[1] * rv_ccf_trimmed[:, 0] + coeff[2]
    mplot.plot(rvs, ccf, 'rp-', label = 'Cross-correlation function (CCF)', 
          linewidth = 30)
    mplot.plot(rv_ccf_trimmed[:,0], fit_ccf, 'bp-', label = 'Parabolic fit', 
          linewidth = 30)
    mplot.xticks(fontsize = 60)
    mplot.yticks(fontsize = 60)
    mplot.xlabel('Radial velocity ($\mathbf{km s^{-1}}$)', fontweight='bold',
          fontsize = 100, labelpad = 100)
    mplot.ylabel('Cross-correlation function', fontweight='bold', 
          fontsize = 100, labelpad = 100)
    mplot.show()


def display_template_spectrum(spectrum, template):   
    mplot.plot(spectrum[:,0], spectrum[:,1], 'grey')
    mplot.plot(template[:,0], template[:,1], 'r')
    mplot.xlabel('Wavelength ($\mathbf{\AA}$)', fontweight='bold',
          fontsize = 100,  labelpad = 100)
    mplot.ylabel('Normalized count', fontweight='bold', 
          fontsize = 100, labelpad = 100)
    mplot.xticks(fontsize = 60)
    mplot.yticks(fontsize = 60)
    mplot.show()

    
def cross_corr_RV(spectrum, template):

    #For correct error estimation, the mean values of template 
    #and observed sepctrum were set to zero.
    
    spectrum[:,1] = np.subtract(spectrum[:,1], np.mean(spectrum[:,1]))
    template[:,1] = np.subtract(template[:,1], np.mean(template[:,1]))
   
    #The squared sums of fluxes for both the template and observed spectrum 
    # are used for rescaling the CCF.
    sigma_s = math.sqrt(sum(np.multiply(spectrum[:, 1], spectrum[:, 1])))
    sigma_t = math.sqrt(sum(np.multiply(template[:, 1], template[:, 1])))
    
    #The CCF is being computed. 
    rvs, ccf = crosscorrRV(spectrum[:, 0], spectrum[:, 1], template[:, 0], 
               template[:, 1], rvmin=RV_MIN, rvmax=RV_MAX, drv=STEP_SIZE)
 
    ccf /= (sigma_s * sigma_t) # Rescaling the CCF
 
    max_rv = rvs[np.argmax(ccf)] #the rv at the peak of CCF

    rv_ccf =  np.zeros((len(rvs),2))
    rv_ccf[:,0] = rvs
    rv_ccf[:,1] = ccf
    rv_ccf_trimmed = rv_ccf[np.logical_and(rv_ccf[:, 0] <= 
                     PRB_FIT_HALF_WINDOW + max_rv, 
                     rv_ccf[:, 0] >= - PRB_FIT_HALF_WINDOW + max_rv), :]
    
    #Parabolic fit is done here. 
    coeff = np.polyfit(rv_ccf_trimmed[:, 0], rv_ccf_trimmed[:, 1], deg=2)   
    '''The line below can be run to plot the ccf with the parabolic fit.''' 
    #display_parabolicfit(coeff, rvs, ccf, rv_ccf_trimmed)
    
    measured_RV = - coeff[1]/(2*coeff[0]) #vertex of the parabola
    
    n , _ = template.shape # Number of bins utilised in cross-correlation
    #The formula from Zucker (2003) is implemented here.
    error = math.sqrt(((np.max(ccf))**2 - 1) 
            /(n * 2 * coeff[0] * np.max(ccf))) 
    # 2 * coeff[0] is the second derivative of the parabolic fit
    return  measured_RV, error


'''Main body of the program starts here'''
#The names of 159 spectra are loaded to a string array
file_names = np.loadtxt('stars_159.txt', str, unpack=False) 

for line_count in range(2,5): # CC2, CC3 and CC4 are implemented. 

    #The rest wavelength list is chosen according
    #to the number of lines utilised.
    rest_wvs = REST_WV_LIST[str(line_count)] 
    
    #The text files for measured RV and errors are created.
    rv_file= open('CC' + str(line_count)+'_RVs.txt', "w")
    rv_err_file= open('CC' + str(line_count)+'_RVs_err.txt', "w")

    #The template fluxes and wavelengths are loaded to a 2D array.
    template_file = open('template_' + 
                    str(line_count)+'lines.txt', mode = 'r') 
    template = cm.file_to_array_txt(template_file)
    
    #Each spectrum is loaded to a 2D array. The RV and
    #associated error are computed. 
    for j in range(0,len(file_names)): 
        
        spectrum = cm.read_fits_file(file_names[j])
   
        cm.preprocess(spectrum)
        #The spectrum is trimmed as it must be shorter from the template 
        #from both edges. 
        spectrum = spectrum[ np.logical_and(spectrum[:,0] > 
                   (rest_wvs[-1] - EDGE_SIZE),
                   spectrum[:,0] < (rest_wvs[0] + EDGE_SIZE)) ,:]
    
        '''The line below can be executed to plot the template and spectrum
        #on the same graph.'''
        #display_template_spectrum(spectrum, template)
   
        rv,error = cross_corr_RV(spectrum, template)
   
        rv_file.write( str(np.round(rv, 4))+"\n")
        rv_err_file.write(str(np.round(error, 4))+"\n")

    rv_file.close()
    rv_err_file.close()


    




    
   

