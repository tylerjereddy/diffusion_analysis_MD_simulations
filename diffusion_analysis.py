'''
Author: Tyler Reddy

The purpose of this Python module is to provide utility functions for analyzing the diffusion of particles in molecular dynamics simulation trajectories using either linear or anomalous diffusion models.'''

def fit_anomalous_diffusion_data(time_data_array,MSD_data_array,axis,species_label,species_type):
    '''This function should fit anomalous diffusion data to Equation 1 in Kneller et al. (2011) J Chem Phys 135: 141105. 
    Input data should include arrays of MSD and time values.'''
    def function_to_fit(time,fractional_diffusion_coefficient,scaling_exponent):
        return 4.0 * fractional_diffusion_coefficient * (time ** scaling_exponent) #equation 1 in the above paper
    optimized_parameter_value_array, estimated_covariance_params_array = scipy.optimize.curve_fit(function_to_fit,time_data_array,MSD_data_array)
    #test plot the resulting fit as well:
    if species_type == 'protein':
        color_dict = {'HA':'red','NA':'blue','M2':'purple'} #to match the linear MSD vs. time data for simpler comparison
    elif species_type == 'lipid':
        color_dict = {'POPS':'blue','DOPE':'red','CHOL':'purple','PPCH':'black','DOPX':'green','FORS':'cyan','DPPC':'grey','DUPC':'brown'}
    test_X = numpy.linspace(time_data_array[0],time_data_array[-1],100)
    test_Y = function_to_fit(test_X, *optimized_parameter_value_array)
    axis.plot(test_X,test_Y,color=color_dict[species_label])
    axis.scatter(time_data_array,MSD_data_array,color=color_dict[species_label],label=species_label)
    return (optimized_parameter_value_array, estimated_covariance_params_array)

def fit_linear_diffusion_data(time_data_array,MSD_data_array,index_window_filter):
    '''The (crude) linear MSD vs. time slope = diffusion constant approach and the error estimate used by g_msd.'''
    x_data_array = time_data_array[index_window_filter:]
    y_data_array = MSD_data_array[index_window_filter:]
    slope, intercept = numpy.polyfit(x_data_array,y_data_array,1)
    diffusion_constant = slope / (4 * 10 ** 7) #convert to standard cm ^ 2 / s units
    #estimate error in D constant using g_msd approach
    first_half_x_data, second_half_x_data = numpy.array_split(x_data_array,2)
    first_half_y_data, second_half_y_data = numpy.array_split(y_data_array,2)
    slope_first_half, intercept_first_half = numpy.polyfit(first_half_x_data,first_half_y_data,1)
    slope_second_half, intercept_second_half = numpy.polyfit(second_half_x_data,second_half_y_data,1)
    diffusion_constant_error_estimate = abs(slope_first_half - slope_second_half) / (4 * 10 ** 7)
    #I've plotted the linear stuff previously so not going to worry about that here, at least for now
    log_x_data_array = numpy.log10(x_data_array)
    log_y_data_array = numpy.log10(y_data_array)
    slope, intercept = numpy.polyfit(log_x_data_array,log_y_data_array,1)
    scaling_exponent = slope
    return (diffusion_constant, diffusion_constant_error_estimate, scaling_exponent)



