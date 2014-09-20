import unittest
import numpy
import numpy.testing
import numpy.random
import scipy
import math
import diffusion_analysis

class Test_anomalous_diffusion_non_linear_fit(unittest.TestCase):

    def setUp(self):
        self.linearly_increasing_time_array = numpy.arange(50)
        self.linear_MSD_array_for_slope_of_two = self.linearly_increasing_time_array * 2
        self.linear_MSD_array_for_slope_of_ten = self.linearly_increasing_time_array * 10
        self.MSD_array_subdiffusive = numpy.copy(self.linear_MSD_array_for_slope_of_two)
        self.MSD_array_subdiffusive[-1] = self.MSD_array_subdiffusive[-1] - (self.MSD_array_subdiffusive[-1] * 0.1) #final point reduced by 10%
        self.MSD_array_superdiffusive = numpy.copy(self.linear_MSD_array_for_slope_of_two)
        self.MSD_array_superdiffusive[-1] = self.MSD_array_superdiffusive[-1] + (self.MSD_array_superdiffusive[-1] * 0.1) #final point increased by 10%

    def tearDown(self):
        del self.linearly_increasing_time_array
        del self.linear_MSD_array_for_slope_of_two
        del self.linear_MSD_array_for_slope_of_ten
        del self.MSD_array_subdiffusive
        del self.MSD_array_superdiffusive

    def test_fractional_diffusion_constant_calculation_linear_data_nonlinear_function(self):
        '''Since linear diffusion is merely a special case of the anomalous diffusion equation, the anomalous diffusion code should return the correct values given linear input data.'''
        for degrees_of_freedom in [1,2,3]: #test for all normal degrees of freedom
            alpha_expected = 1. #input data is perfectly linear
            coefficient = degrees_of_freedom * 2. #2,4,6 for 1D, 2D, 3D, respectively
            # 2 = slope test
            fractional_diffusion_coefficient, standard_deviation_fractional_diffusion_coefficient, alpha, standard_deviation_alpha,sample_fitting_data_X_values_nanoseconds,sample_fitting_data_Y_values_Angstroms = diffusion_analysis.fit_anomalous_diffusion_data(self.linearly_increasing_time_array,self.linear_MSD_array_for_slope_of_two,degrees_of_freedom=degrees_of_freedom)
            D_expected = 2. / coefficient #simple linear D
            self.assertEqual(fractional_diffusion_coefficient,D_expected)
            self.assertEqual(alpha,alpha_expected)
            # 10 = slope test
            fractional_diffusion_coefficient, standard_deviation_fractional_diffusion_coefficient, alpha, standard_deviation_alpha,sample_fitting_data_X_values_nanoseconds,sample_fitting_data_Y_values_Angstroms = diffusion_analysis.fit_anomalous_diffusion_data(self.linearly_increasing_time_array,self.linear_MSD_array_for_slope_of_ten,degrees_of_freedom=degrees_of_freedom)
            D_expected = 10. / coefficient #simple linear D
            self.assertEqual(fractional_diffusion_coefficient,D_expected)
            self.assertEqual(alpha,alpha_expected)
            
    def test_scaling_exponent_trends_nonlinear_function(self):
        '''Ensure that alpha is > 1 when fed superdiffusive data and alpha < 1 when fed subdiffusive data.'''
        for degrees_of_freedom in [1,2,3]:
            calculated_subdiffusive_alpha = diffusion_analysis.fit_anomalous_diffusion_data(self.linearly_increasing_time_array,self.MSD_array_subdiffusive,degrees_of_freedom=degrees_of_freedom)[2]
            self.assertLess(calculated_subdiffusive_alpha,1.0)
            calculated_superdiffusive_alpha = diffusion_analysis.fit_anomalous_diffusion_data(self.linearly_increasing_time_array,self.MSD_array_superdiffusive,degrees_of_freedom=degrees_of_freedom)[2]
            self.assertGreater(calculated_superdiffusive_alpha,1.0)



class Test_linear_fit_normal_diffusion(unittest.TestCase):
        
    def setUp(self):
        self.linearly_increasing_time_array = numpy.arange(220)
        self.random_slope_value = numpy.random.random_sample() * 10.
        self.linear_MSD_array_for_random_slope = self.linearly_increasing_time_array * self.random_slope_value

    def tearDown(self):
        del self.linearly_increasing_time_array
        del self.random_slope_value
        del self.linear_MSD_array_for_random_slope
        
    def test_linear_data_on_linear_diffusion_function(self):
        '''Test the linear fit (normal diffusion) function with linear input data with a random slope value.'''
        for degrees_of_freedom in [1,2,3]: #test for all normal degrees of freedom
            coefficient = degrees_of_freedom * 2. #2,4,6 for 1D, 2D, 3D, respectively
            D_expected = self.random_slope_value / coefficient 
            D_actual = diffusion_analysis.fit_linear_diffusion_data(self.linearly_increasing_time_array,self.linear_MSD_array_for_random_slope,degrees_of_freedom=degrees_of_freedom)[0]
            numpy.testing.assert_almost_equal(D_actual,D_expected,decimal=10)


