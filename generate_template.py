import common_methods as cm
import numpy as np
import matplotlib.pyplot as mplot
import pyspeckit


EDGE_SIZE = 50
GAUSSIAN_HALF_WINDOW = 10 # The Gaussian fits are created within 20 A window.
GAUSSIAN_WIDTH = 2.5 # The width of each Gaussian fit in all templates

REST_WV_LIST = {
    "2": [8750.472, 8665.018],
    "3": [8750.472, 8665.018, 8598.392],
    "4": [8750.472, 8665.018, 8598.392, 8545.383]}

'''Writes the template array to a text file.'''
def array_to_file(output_file_name, spectrum_data):
    x, y = spectrum_data.shape
    spectrum_data = np.round(spectrum_data, 4)

    output_file = open(output_file_name, "w")
    for i in range(0, x):
        output_file.write(
        str(spectrum_data[i][0])+" " + str(spectrum_data[i][1])+"\n")

    output_file.close()

''' Used to plot the template''' 
def plot_template(template):
    mplot.plot(template[:, 0], template[:, 1], 'r')
    mplot.xlabel('Wavelength ($\AA$)', fontweight='bold',fontsize=100)
    mplot.ylabel('Count', fontweight='bold', fontsize=100)
    mplot.xticks(fontsize = 70)
    mplot.yticks(fontsize = 70)
    mplot.show()

'''From xdata, creates y-data for Gaussian curves''' 
def gauss(x, a, x0, sigma):
    d = (x-x0)/sigma
    return (a)*np.exp(-d*d/2)

'''The backbone of template generation process'''
def generate_template(spectrum, line_count, rest_wvs):
    sp = pyspeckit.Spectrum(data=spectrum[:, 1], xarr=spectrum[:, 0],
         xarrkwargs={'unit': 'Angstrom'})
    guesses, fixed = set_initial_parameters(rest_wvs, line_count)
    sp.specfit.multifit('gaussian', guesses=guesses, 
      fixed = fixed, 
       renormalize='auto', annotate=None, composite_lw=5, lw=5)
    
    parameters = np.reshape(sp.specfit.modelpars , (line_count, 3))

    for j in range(0, line_count):
        parameters[j][1] = rest_wvs[j]
    template = np.copy(spectrum) # Template bin size is identical to the 
    # observed spectra. Its size is 25 A longer from both edges. The observed 
    #spectra will be truncated before cross-correlation

    template[:, 1] = 0.0

    for i in range(0, line_count):
        max_value = parameters[i][1] + GAUSSIAN_HALF_WINDOW
        min_value = parameters[i][1] - GAUSSIAN_HALF_WINDOW
        segment = template[np.logical_and(
            template[:, 0] <= max_value, template[:, 0] >= min_value), :]
        segment[:, 1] = gauss(segment[:, 0], parameters[i, 0],
                              parameters[i, 1], parameters[i, 2])
        index_max = np.where(template[:, 0] == segment[-1][0])
        index_min = np.where(template[:, 0] == segment[0][0])

        template[index_min[0][0]:index_max[0][0]+1, 1] = segment[:, 1]
  
    template = template[ np.logical_and(template[:,0] > 
               (rest_wvs[-1] - EDGE_SIZE),
               template[:,0] < (rest_wvs[0] + EDGE_SIZE)) ,:]

    plot_template(template)
    array_to_file('template_' + str(line_count)+'lines.txt', template)
    
'''Initial Gaussian parameters to have a better optimisation'''
def set_initial_parameters(rest_wvs, line_count):

    guesses = []
    fixed = []
   
    if(line_count == 2):
        guesses = [-0.4, rest_wvs[0], GAUSSIAN_WIDTH, -
                   0.3, rest_wvs[1], GAUSSIAN_WIDTH]      
        fixed = [False, False, True, False, False, True] 

    elif(line_count == 3):
        guesses = [-0.4, rest_wvs[0], GAUSSIAN_WIDTH,
                   -0.3, rest_wvs[1], GAUSSIAN_WIDTH,
                   -0.2, rest_wvs[2], GAUSSIAN_WIDTH]      
        fixed = [False, False, True, False, False,True, False, False,True] 

    elif(line_count == 4):
        guesses = [-0.4, rest_wvs[0], GAUSSIAN_WIDTH,
                   -0.3, rest_wvs[1], GAUSSIAN_WIDTH,
                   -0.2, rest_wvs[2], GAUSSIAN_WIDTH,
                   -0.1, rest_wvs[3], GAUSSIAN_WIDTH]     
        fixed = [False, False, True, False, False, True, 
                 False, False, True, False, False,True] 

    return guesses, fixed

    
'''Main body starts here'''
spectrum = cm.read_fits_file('02325.fits')
cm.preprocess(spectrum)

for line_count in range(2,5):#4-line, 3-line and 2-line templates are created. 
    rest_wvs  =  REST_WV_LIST[str(line_count)] 
    generate_template(spectrum, line_count, rest_wvs )



