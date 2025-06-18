import unittest
import pandas as pd
from pathlib import Path
from calibration_curves.hplc_urea import exponentialfit, linearfit

class TestExponentialFit(unittest.TestCase):
    def test_linear_fit_taylor_example(self):
        # use the data from Taylor Error Analysis linear fit example (page 184)
        p = Path('./tests/taylor_linear_example.xlsx')
        y, dy, res = linearfit(p, x='x', y='y')
        a1 = res.params.iloc[1]
        self.assertAlmostEqual(a1, 2.06, places=2)


    def test_exponential_fit_taylor_example(self):
        # use the data from Taylor Error Analysis exponential fit example (page 195)
        p = Path('./tests/taylor_exponential_example.xlsx')
        y, dy, res = exponentialfit(p, x='x', y='y')

        a0 = res.params.iloc[0]
        self.assertAlmostEqual(a0, 11.93, places=2)


if __name__ == '__main__':
    unittest.main()
