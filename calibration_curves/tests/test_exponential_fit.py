import unittest
import pandas as pd
from pathlib import Path
from calibration_curves.hplc_urea import exponentialfit

class TestExponentialFit(unittest.TestCase):
    def test_against_taylor_example(self):
        p = Path('./tests/taylor_exponential_example.xlsx') # use the data from Taylor Error Analysis exponential fit example

        # taylor_data = pd.read_excel(p)
        y, dy, res = exponentialfit(p, x='x', y='y')
        a0 = res.params.iloc[0]
        a1 = res.params.iloc[1]

        self.assertAlmostEqual(a0, 11.93, places=2)


if __name__ == '__main__':
    unittest.main()
