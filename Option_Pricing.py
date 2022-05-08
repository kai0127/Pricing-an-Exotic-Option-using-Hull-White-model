#  Copyright (c) 2022/5/8 上午1:04 Last Modified By Kaiwen Zhou

"""
This file contains methods and functions by which we can run the Monte Carlo Simulation as many times as you want and
give an approximation for the final price of the Exotic Option by calculating the cumulative average of all simulated
results. Also, we plot the simulated process of domestic short rate r_d and the cumulative average of the prices of the
Exotic Option we get from MC simulations.

@created 05/04/2022 - 2:24 PM
@author Kaiwen Zhou
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Bond_Data import Bond_Data
from Fed_Data import Fed_Data
from SIngle_time_Monte_Carlo import Single_time_Monte_Carlo
from Nikkei_Data import Nikkei_Data


# 1. Bond data
# the url of the treasury bond yields provided by the treasury.gov
url = 'https://home.treasury.gov/sites/default/files/interest-rates/yield.xml'
# Initialize the Bond_Data object bond_data
bond_data = Bond_Data(url)
# Now specify the knots (Note that the more knots we have, the better the approximation)
x_indexes = np.linspace(0, 30, 360)  # 0 to 30 years, 360 ticks


# 2. Federal Funds Rate
api_key = '9a60c60d8e0a0ec28887d3ec5b8d0dd4'
fred_data = Fed_Data(api_key = api_key)
current_federal_funds_rate = fred_data.get_current_federal_funds_rate()
#current_domestic_risk_free_rate = bond_data.yields_in_percentage[2] /100  # 3 month treasury bill rate


# 3. Nikkei Data
nikkei_data = Nikkei_Data()
nikkei_dividends = nikkei_data.get_latest_dividend_yield() # Nikkei dividends rate
current_nikkei_price = fred_data.get_latest_daily_average_nikkei_price()  # current Nikkei price

# 4. Conditions given by the Exotic option
maturity_T = 3  # maturity of the exotic option
Delta = 0.25  # a 3 months period
strike_price_nikkei = 1  # strike price for the nikkei part
strike_price_libor = 1  # strike price for the libor part

# 5. Hull-White model dr_t = (theta_t-a*r_t)dt + sigma_t dW
domestic_risk_free_rate_vol = 1.5 / 100  # volatility for the USD risk-free rate

mean_reversion_speed = 0.03  # mean-reversion speed for the USD risk-free rate (normally practitioners use 0.03 or 0.05)
foreign_risk_free_rate = -0.1 / 100  # JPY short rate
foreign_asset_vol = 24.78 / 100  # Nikkei price implied volatility
exchange_rate_vol = 9.73 / 100  # JPY/USD exchange rate implied volatility

correlation_foreign_asset_and_exchange_rate = fred_data.get_correlation_foreign_asset_and_exchange_rate()
correlation_foreign_asset_and_domestic_risk_free_rate = 0.3


# 6. Monte Carlo
iteration_times = 1000  # decides how many times of iterations for a single run of Monte Carlo


"""!!!!MONTE-CARLO!!!!"""
single_time_monte_carlo = Single_time_Monte_Carlo(
                                                  bond_data=bond_data,
                                                  domestic_risk_free_rate_vol=domestic_risk_free_rate_vol,
                                                  mean_reversion_speed=mean_reversion_speed,
                                                  maturity_T=maturity_T,
                                                  Delta=Delta,
                                                  strike_price_nikkei=strike_price_nikkei,
                                                  strike_price_libor=strike_price_libor,
                                                  iteration_times=iteration_times,
                                                  current_federal_funds_rate=current_federal_funds_rate,
                                                  current_nikkei_price=current_nikkei_price,
                                                  correlation_foreign_asset_and_domestic_risk_free_rate=correlation_foreign_asset_and_domestic_risk_free_rate,
                                                  correlation_foreign_asset_and_exchange_rate=correlation_foreign_asset_and_exchange_rate,
                                                  foreign_risk_free_rate=foreign_risk_free_rate,
                                                  nikkei_dividends=nikkei_dividends,
                                                  foreign_asset_vol=foreign_asset_vol,
                                                  exchange_rate_vol=exchange_rate_vol
                                                  )
# Setting up for Monte Carlo
NUM_SIMULATION = 1000  # num of Monte Carlo simulations
df_domestic_short_rate_processes = pd.DataFrame()  # dataframe storing r_d process for each simulation
list_of_prices_Exotic_option = []  # list storing the price of the Exotic Option for each simulation

# Start Simulating
for i in range(NUM_SIMULATION):
    single_time_monte_carlo.start_monte_carlo()  # single run
    # store the simulated r_d process in the dataframe
    df_domestic_short_rate_processes[i] = single_time_monte_carlo.domestic_risk_free_rate_list
    # store the price of the Exotic Option in list_of_prices_Exotic_option
    list_of_prices_Exotic_option.append(single_time_monte_carlo.price_Exotic_Option_at_maturity())


def cumulative_average_of_list(listt):
    """
    Calculate the cumulative average of the given list
    :param listt: list of numbers
    :return: list of cumulative average of the given list
    """
    list_cumulative_average = [sum(listt[:i + 1]) / (i + 1) for i in range(len(listt))]
    return list_cumulative_average


# compute the cumulative average of Exotic option's prices
cumulative_average_of_Exotic_Option_prices = cumulative_average_of_list(list_of_prices_Exotic_option)

# 7. Print the approximated price of the Exotic Option
print('The price of the Exotic Option is approximately (in USD): ', cumulative_average_of_Exotic_Option_prices[NUM_SIMULATION-1])

# 8.PLOT
fig, (ax1,ax2) = plt.subplots(2)
ax1.plot(single_time_monte_carlo.timeline, df_domestic_short_rate_processes)
ax1.set_title("xlabel = years")
ax1.set_ylabel('domestic short rate')
ax2.plot(cumulative_average_of_Exotic_Option_prices)
ax2.set_xlabel('number of simulations')
ax2.set_ylabel('cumulative average of \n simulated Exotic Option prices')
plt.show()
