#  Copyright (c) 2022/5/7 下午10:29 Last Modified By Kaiwen Zhou

"""
This module/class contains methods and functions by which we can find data related to bond such as, yield curve, bond prices,
instantaneous forward rate curve, and the curve for the first derivative of the instantaneous forward rate
i.e. y, p(0,T), f(0,T) and f_T(0,T)

@created 05/04/2022 - 2:24 PM
@author Kaiwen Zhou
"""
import numpy as np
from scipy import interpolate
import xmltodict  # This package is being used for translate .xml data to OrderedDict
import requests  # This package is being used for accessing url


def get_data_from_url(url):
    """
    Extract raw data from the url for further use
    :param url: the url of the treasury bond yields provided by the treasury.gov
    :return: full data provided by the treasury on yields
    """
    # get data from the given url
    data = requests.get(url).content

    # Translate .xml data to Python OrderedDict data
    data = xmltodict.parse(data)
    return data


def get_data_for_newest_day(url):
    """
    Get the data for the nearest day preceding today (inclusive)
    :param url: the url of the treasury bond yields provided by the treasury.gov
    :return: the data for treasury yields on the nearest day preceding today (inclusive)
    """
    # get data from the given url
    data = get_data_from_url(url)

    # This for loop is being used for finding data for the latest week
    for i in range(4, -1, -1):
        try:
            last_week_of_the_month_data = data['QR_BC_CM']['LIST_G_WEEK_OF_MONTH']['G_WEEK_OF_MONTH'][i]
            break
        except:
            # If for i= 4,3,2,1,0, we all have exceptions, that means  last_week_of_the_month_data  is
            # not stored in a list, and it has only data for that specific day.
            last_week_of_the_month_data = data['QR_BC_CM']['LIST_G_WEEK_OF_MONTH']['G_WEEK_OF_MONTH']
            continue
    # This try-except block is being used for extracting data for the lastest day
    try:
        lastest_day_of_the_month_data = last_week_of_the_month_data['LIST_G_NEW_DATE']['G_NEW_DATE'][-1]
    except:
        lastest_day_of_the_month_data = last_week_of_the_month_data['LIST_G_NEW_DATE']['G_NEW_DATE']

    return lastest_day_of_the_month_data


# Put all useful information in a list
# NOTE: This function is highly related to the function get_data_for_newest_day(data) above, i.e.
# You will have to input the return value of the function get_data_for_newest_day(data)
def put_all_data_in_list(data):
    """

    :param data: !!!result of the function get_data_for_newest_day(url)!!!
    :return: all yields in list of strings and dates
    """
    list1 = []
    list1.append(data['BID_CURVE_DATE'])
    list1.append(data['DAY_OF_WEEK'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_1MONTH'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_2MONTH'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_3MONTH'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_6MONTH'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_1YEAR'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_2YEAR'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_3YEAR'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_5YEAR'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_7YEAR'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_10YEAR'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_20YEAR'])
    list1.append(data['LIST_G_BC_CAT']['G_BC_CAT']['BC_30YEAR'])
    return list1


# If you want to return only the yield data, use the simplified version (function below)
def only_yields_in_percentage(data):
    """
    Get all yields of treasury bonds with different maturities
    :param data: !!!result of the function get_data_for_newest_day(url)!!!
    :return: all yields in list of floats
    """
    list1 = put_all_data_in_list(data)
    yields_strings = list1[2:]
    yields_floats = [float(i) for i in yields_strings]
    return yields_floats


class Bond_Data(object):

    def __init__(self, url):
        self.url = url
        self.data = get_data_for_newest_day(url)
        self.yields_in_percentage = only_yields_in_percentage(self.data)
        self.years = [1/12, 2/12, 3/12, 6/12, 12/12, 24/12, 36/12, 60/12, 84/12, 120/12, 240/12, 360/12]
        self.bond_prices = self.get_bond_prices()
        # generate the piecewise cubic (bounded by k = 3) spline; s=0 is for smoothness (normally zero)
        self.tck_bond = interpolate.splrep(self.years, self.bond_prices, s=0, k=3)  # The interpolation has
        # completed here!
        self.date = self.get_date()

    def get_date(self):
        """
        Get the date for the obtained data
        :return: the date for the obtained  data
        """
        return put_all_data_in_list(self.data)[0]

    def get_bond_prices(self):
        """
        Get the raw bond prices observed in the markets

        !!!WARNING: Here we are naively using the yields of coupon bond to compute the price of zero coupon bonds,
        specifically for bonds with maturity over 1 year. For bonds with maturity less or equal to 1 year (treasury bills)
        this approach is actually correct.!!!

        :return: market bond prices
        """
        # generate all prices for zero coupon bonds with different maturities
        bond_prices = [np.exp(-self.yields_in_percentage[i] / 100 * self.years[i] ) for i in range(12)]
        return bond_prices

    def get_value_on_cubic_spline_for_bond_prices(self, x_indexes):
        """
        Return the Cubic Spline interpolation of the ture bond prices (raw prices observed in markets)
        :rtype: object
        """
        bond_prices_approx = interpolate.splev(x_indexes, self.tck_bond, der=0)
        return bond_prices_approx

    def get_value_for_forward_rate(self, x_indexes):
        """
        The instantaneous forward rate = -1/p(0,T) * (dp(0,T)/dT)
        :param x_indexes: cubic spline interpolation knots
        :return: the values of forward rates (annualized & not in percentage)
        """
        bond_prices_approx = interpolate.splev(x_indexes, self.tck_bond, der=0)
        first_derivative_bond_prices_approx = interpolate.splev(x_indexes, self.tck_bond, der=1)
        return - first_derivative_bond_prices_approx / bond_prices_approx

    def get_value_for_T_derivative_of_forward_rate(self, x_indexes):
        """
        Directly use  formula
        the derivative of the forward rate = -1/p(0,T) * (d^2p(0,T)/dT^2) + 1/p^2(0,T) * (dp(0,T)/dT) * (dp(0,T)/dT)
        :param x_indexes: cubic spline interpolation knots
        :return: the first derivative of the instantaneous forward rate induced by the cubic spline interpolation of
        the market bond prices w.r.t. knots specified by x_indexes (annualized & not in percentage)
        """
        bond_prices_approx = interpolate.splev(x_indexes, self.tck_bond, der=0)
        first_derivative_bond_prices_approx = interpolate.splev(x_indexes, self.tck_bond, der=1)
        second_derivative_bond_prices_approx = interpolate.splev(x_indexes, self.tck_bond, der=2)
        return -1 / bond_prices_approx * second_derivative_bond_prices_approx \
               + 1 / bond_prices_approx ** 2 * first_derivative_bond_prices_approx * first_derivative_bond_prices_approx
