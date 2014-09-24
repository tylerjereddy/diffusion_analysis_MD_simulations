'''
Author: Tyler Reddy

The purpose of this Python module is to provide utility functions for analyzing the diffusion of particles in molecular dynamics simulation trajectories using either linear or anomalous diffusion models.'''

import numpy
import scipy
import scipy.optimize

def fit_anomalous_diffusion_data(time_data_array,MSD_data_array,degrees_of_freedom=2):
    '''This function should fit anomalous diffusion data to Equation 1 in [1]_, and return appropriate diffusion parameters.
    
    An appropriate coefficient will be assigned based on the specified `degrees_of_freedom`. The latter value defaults to 2 (i.e., a planar phospholipid bilayer).
    Input data should include arrays of MSD (in Angstroms) and time values (in ns).
    The results are returned in a tuple.

    Parameters
    ----------
    time_data_array : array_like
        Input array of time window sizes (nanosecond units)
    MSD_data_array : array_like
        Input array of MSD values (Angstrom units; order matched to time_data_array)
    degrees_of_freedom : int
        The degrees of freedom for the diffusional process (1, 2 or 3; default 2)

    Returns
    -------
    fractional_diffusion_coefficient
        The fractional diffusion coefficient (units of Angstrom ** 2 / ns ** alpha)
    standard_deviation_fractional_diffusion_coefficient
        The standard deviation of the fractional diffusion coefficent (units of Angstrom ** 2 / ns ** alpha)
    alpha
        The scaling exponent (no dimensions) of the non-linear fit
    standard_deviation_alpha
        The standard deviation of the scaling exponent (no dimensions)
    sample_fitting_data_X_values_nanoseconds
        An array of time window sizes (x values) that may be used to plot the non-linear fit curve
    sample_fitting_data_Y_values_Angstroms
        An array of MSD values (y values) that may be used to plot the non-linear fit curve

    Raises
    ------
    ValueError
        If the time window and MSD arrays do not have the same shape

    References
    ----------

    .. _[1] Kneller et al. (2011) J Chem Phys 135: 141105.


    Examples
    --------
    Calculate fractional diffusion coefficient and alpha from artificial data (would typically obtain empirical data from an MD simulation trajectory):

    >>> import diffusion_analysis
    >>> import numpy
    >>> artificial_time_values = numpy.arange(10)
    >>> artificial_MSD_values = numpy.array([0.,1.,2.,2.2,3.6,4.7,5.8,6.6,7.0,6.9])
    >>> results_tuple = diffusion_analysis.fit_anomalous_diffusion_data(artificial_time_values,artificial_MSD_values)
    >>> D, D_std, alpha, alpha_std = results_tuple[0:4]
    >>> print D, D_std, alpha, alpha_std 
    0.268426206526 0.0429995249239 0.891231967011 0.0832911559401

    Plot the non-linear fit data:

    >>> import matplotlib
    >>> import matplotlib.pyplot as plt
    >>> sample_fit_x_values, sample_fit_y_values = results_tuple[4:]
    >>> p = plt.plot(sample_fit_x_values,sample_fit_y_values,'-',artificial_time_values,artificial_MSD_values,'.')

    .. image:: example_nonlinear.png

        '''

    if time_data_array.shape != MSD_data_array.shape:
        raise ValueError("The shape of time_data_array must match the shape of MSD_data_array.")

    def function_to_fit(time,fractional_diffusion_coefficient,scaling_exponent):
        coefficient_dictionary = {1:2,2:4,3:6} #dictionary for mapping degrees_of_freedom to coefficient in fitting equation
        coefficient = coefficient_dictionary[degrees_of_freedom]
        return coefficient * fractional_diffusion_coefficient * (time ** scaling_exponent) #equation 1 in the above paper with appropriate coefficient based on degrees of freedom

    #fit the above function to the data and pull out the resulting parameters
    optimized_parameter_value_array, estimated_covariance_params_array = scipy.optimize.curve_fit(function_to_fit,time_data_array,MSD_data_array)
    #generate sample fitting data over the range of time window values (user can plot if they wish)
    sample_fitting_data_X_values_nanoseconds = numpy.linspace(time_data_array[0],time_data_array[-1],100)
    sample_fitting_data_Y_values_Angstroms = function_to_fit(sample_fitting_data_X_values_nanoseconds, *optimized_parameter_value_array)
    #could then plot the non-linear fit curve in matplotlib with, for example: axis.plot(sample_fitting_data_X_values_nanoseconds,sample_fitting_data_Y_values_Angstroms,color='black')
    #could plot the original data points alongside the fit (MSD vs time) with, for example: axis.scatter(time_data_array,MSD_data_array,color='black')

    #extract pertinent values from the scipy curve_fit arrays (D_alpha, alpha, and their standard deviations)
    parameter_standard_deviation_array = numpy.sqrt(numpy.diagonal(estimated_covariance_params_array))
    fractional_diffusion_coefficient = optimized_parameter_value_array[0]
    standard_deviation_fractional_diffusion_coefficient = parameter_standard_deviation_array[0]
    alpha = optimized_parameter_value_array[1]
    standard_deviation_alpha = parameter_standard_deviation_array[1]

    return (fractional_diffusion_coefficient, standard_deviation_fractional_diffusion_coefficient, alpha, standard_deviation_alpha,sample_fitting_data_X_values_nanoseconds,sample_fitting_data_Y_values_Angstroms)

def fit_linear_diffusion_data(time_data_array,MSD_data_array,degrees_of_freedom=2):
    '''The linear (i.e., normal, random-walk) MSD vs. time diffusion constant calculation.
    
    The results are returned in a tuple.

    Parameters
    ----------
    time_data_array : array_like
        Input array of time window sizes (nanosecond units)
    MSD_data_array : array_like
        Input array of MSD values (Angstrom units; order matched to time_data_array)
    degrees_of_freedom : int
        The degrees of freedom for the diffusional process (1, 2 or 3; default 2)

    Returns
    -------
    diffusion_constant
        The linear (or normal, random-walk) diffusion coefficient (units of Angstrom ** 2 / ns ** alpha)
    diffusion_constant_error_estimate
        The estimated uncertainty in the diffusion constant (units of Angstrom ** 2 / ns ** alpha), calculated as the difference in the slopes of the two halves of the data. A similar approach is used by GROMACS g_msd.
    sample_fitting_data_X_values_nanoseconds
        An array of time window sizes (x values) that may be used to plot the linear fit 
    sample_fitting_data_Y_values_Angstroms
        An array of MSD values (y values) that may be used to plot the linear fit 

    Raises
    ------
    ValueError
        If the time window and MSD arrays do not have the same shape
    '''

    if time_data_array.shape != MSD_data_array.shape:
        raise ValueError("The shape of time_data_array must match the shape of MSD_data_array.")

    coefficient_dictionary = {1:2.,2:4.,3:6.} #dictionary for mapping degrees_of_freedom to coefficient in fitting equation
    coefficient = coefficient_dictionary[degrees_of_freedom]
    
    x_data_array = time_data_array
    y_data_array = MSD_data_array
    z = numpy.polyfit(x_data_array,y_data_array,1)
    slope, intercept = z
    diffusion_constant = slope / coefficient 

    #estimate error in D constant using g_msd approach:
    first_half_x_data, second_half_x_data = numpy.array_split(x_data_array,2)
    first_half_y_data, second_half_y_data = numpy.array_split(y_data_array,2)
    slope_first_half, intercept_first_half = numpy.polyfit(first_half_x_data,first_half_y_data,1)
    slope_second_half, intercept_second_half = numpy.polyfit(second_half_x_data,second_half_y_data,1)
    diffusion_constant_error_estimate = abs(slope_first_half - slope_second_half) / coefficient 

    #use poly1d object for polynomial calling convenience (to provide plotting fit data for user if they want to use it):
    p = numpy.poly1d(z)
    sample_fitting_data_X_values_nanoseconds = numpy.linspace(time_data_array[0],time_data_array[-1],100)
    sample_fitting_data_Y_values_Angstroms = p(sample_fitting_data_X_values_nanoseconds)

    return (diffusion_constant, diffusion_constant_error_estimate,sample_fitting_data_X_values_nanoseconds,sample_fitting_data_Y_values_Angstroms)



