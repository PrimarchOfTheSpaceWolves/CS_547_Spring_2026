import unittest
import General_Testing as GT
import numpy as np
import A02
from Test_A02 import *

ground_kernels_sep = [
        True, 
        True, 
        True,
        False,
        True,
        True,
        False,
        True,
        True   
]

class Test_A02_Grad(Test_A02):  
    
    def perform_test_check_if_separable(self, kernel_index):
        kernel = ground_kernels[kernel_index]
        separable, vert_filter, horiz_filter = A02.check_if_separable(kernel=kernel)
        
        ground_sep = ground_kernels_sep[kernel_index]
        if ground_sep:
            msg = "Kernel IS separable"
        else:
            msg = "Kernel IS NOT separable"
            
        self.assertEqual(separable, ground_kernels_sep[kernel_index], msg)
        
        if separable and ground_sep:
            recon_filter = vert_filter @ horiz_filter
            GT.check_for_unequal("Separable vectors for filter " + str(kernel_index) + " not correct!", 
                             self.all_filters[kernel_index], 
                             recon_filter, kernel)
            
    def test_check_if_separable(self):
        for i in range(len(self.all_filters)):
            with self.subTest(filename=self.all_filters[i]):
                self.perform_test_check_if_separable(i)
        
    def test_do_convolution_separable(self):
        for i in range(len(self.all_filters)):       
            if ground_kernels_sep[i]:                 
                for j in range(len(self.all_filenames)):                
                    with self.subTest(filter=self.all_filters[i], image=self.all_filenames[j]):
                        self.perform_test_do_convolution_one_image(i, j, A02.do_convolution_separable)  
            else:
                self.assertIsNone(A02.do_convolution_separable(np.zeros((100,100), dtype="uint8"), 
                                                               ground_kernels[i]))  
            
        
    def test_do_convolution_optimal(self):
        self.perform_test_do_convolution(A02.do_convolution_optimal)
                               
def main():
    runner = unittest.TextTestRunner()
    test_cases = unittest.TestLoader().loadTestsFromTestCase(Test_A02_Grad)    
    runner.run(test_cases)

if __name__ == '__main__':    
    main()

