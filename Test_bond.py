#  Copyright (c) 2022/5/5 下午12:20 Last Modified By Kaiwen Zhou

"""
This is not technically a unit-test for object class Bond_Data. It works as a sanity check for the necessary data like
market bond prices, yields in percentage, instantaneous forward rate, and the derivative of the instantaneous forward
rate. And each test function visualizes one data among those mentioned above via matplotlib

@created 05/04/2022 - 2:24 PM
@author Kaiwen Zhou
"""
import unittest
import matplotlib.pyplot as plt
import numpy as np

from Bond_Data import Bond_Data


# the url of the treasury bond yields provided by the treasury.gov
url = 'https://home.treasury.gov/sites/default/files/interest-rates/yield.xml'
# Initialize the Bond_Data object bond_data
bond_data = Bond_Data(url)

# Now specify the knots (Note that the more knots we have, the better the approximation)
x_indexes = np.linspace(0, 30, 360)  # 0 to 30 years, 360 ticks


class MyTestCase(unittest.TestCase):

    def test_raw_yields_in_percentage(self):
        """
        PLOT the yields_in_percentage with respect to 12 corresponding maturities
        By Kaiwen Zhou
        """
        # extract yields_in_percentage attributes in object bond_data
        yields_in_percentage = bond_data.yields_in_percentage

        # PLOT
        plt.figure()
        plt.xlabel('years')
        plt.ylabel('yields')
        # plot yields in percentage w.r.t. months
        plt.plot(bond_data.years, yields_in_percentage)
        plt.legend(['Yield Curve ' + bond_data.date])
        plt.title('Yield Curve ' + bond_data.date)
        plt.show()

    def test_market_bond_prices(self):
        """
        PLOT the market bond prices with respect to 12 corresponding maturities
        and  the cubic spline approximation of the market bond prices w.r.t. 360 knots specified by x_indexes
        """
        # extract yields_in_percentage attributes in object bond_data
        bond_prices = bond_data.bond_prices

        # Use splev() to  evaluate the spline
        bond_prices_cubic_spline = bond_data.get_value_on_cubic_spline_for_bond_prices(x_indexes)

        # PLOT
        plt.figure()
        plt.xlabel('years')
        plt.ylabel('zero coupon bonds prices')
        # Here (x_indexes, bond_prices_cubic_spline) is the cubic spline interpolation
        # (bond_data.months, bond_prices, 'r') is the curve of original function painted in blue
        plt.plot(x_indexes, bond_prices_cubic_spline, bond_data.years, bond_prices, 'r')
        plt.legend(['Cubic Spline Approximation', 'zero coupon bonds prices ' + bond_data.date])
        plt.title('Cubic-spline interpolation ' + bond_data.date)
        plt.show()

    def test_induced_forward_rates(self):
        """
        PLOT instantaneous forward rate induced by the cubic spline interpolation of the market bond prices
        w.r.t. 360 knots specified by x_indexes
        """
        # Get instantaneous forward rate induced by the cubic spline interpolation of the market bond prices
        induced_forward_rates = bond_data.get_value_for_forward_rate(x_indexes)

        # PLOT
        plt.figure()
        plt.xlabel('years')
        plt.ylabel('instantaneous forward rates')
        plt.plot(x_indexes, induced_forward_rates)
        plt.legend(['induced instantaneous forward rate' + bond_data.date])
        plt.title('Zero coupon bonds prices induced instantaneous forward rate' + bond_data.date)
        plt.show()

    def test_derivative_of_forward_rates(self):
        """
        PLOT the first derivative of the instantaneous forward rate induced by the cubic spline interpolation of
        the market bond prices w.r.t. 360 knots specified by x_indexes
        """
        # Get the first derivative of the instantaneous forward rate induced by the cubic spline interpolation of
        # the market bond prices

        the_first_derivative_of_induced_forward_rate = bond_data.get_value_for_T_derivative_of_forward_rate(x_indexes)

        # PLOT
        plt.figure()
        plt.xlabel('years')
        plt.ylabel('the first derivative of the instantaneous forward rate')
        plt.plot(x_indexes, the_first_derivative_of_induced_forward_rate)
        #plt.legend(['the first derivative of the instantaneous forward rate '+ bond_data.date])
        plt.title('the first derivative of the instantaneous forward rate ' + bond_data.date)
        plt.show()


if __name__ == '__main__':
    unittest.main()
