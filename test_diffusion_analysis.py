import unittest
import numpy
import numpy.testing
import scipy
import math
import diffusion_analysis

class Test_anomalous_diffusion_non_linear_fit(unittest.TestCase):

    def setUp(self):
        self.linearly_increasing_time_array = numpy.arange(50)
        self.linear_MSD_array_for_slope_of_two = self.linearly_increasing_time_array * 2
        self.linear_MSD_array_for_slope_of_ten = self.linearly_increasing_time_array * 10

    def tearDown(self):
        del self.linearly_increasing_time_array
        del self.linear_MSD_array_for_slope_of_two
        del self.linear_MSD_array_for_slope_of_ten

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


        

