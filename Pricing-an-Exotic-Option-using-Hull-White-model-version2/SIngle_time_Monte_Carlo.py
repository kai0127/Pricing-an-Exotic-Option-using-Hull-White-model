#  Copyright (c) 2022/5/8 下午11:22 Last Modified By Kaiwen Zhou

"""
This module/class contains methods and functions by which we can run the Monte Carlo Simulation on domestic short rate
and foreign asset price process for a SINGLE time, and calculate the final approximation for the price of the Exotic
Option.

r_d: domestic short rate, S^f: foreign asset price process

@created 05/04/2022 - 2:24 PM
@author Kaiwen Zhou
"""
import numpy as np


class Single_time_Monte_Carlo(object):

    def __init__(self,
                 bond_data=None,
                 domestic_risk_free_rate_vol=None,
                 mean_reversion_speed=None,
                 maturity_T=None,
                 Delta=None,
                 strike_price_nikkei=None,
                 strike_price_libor=None,
                 iteration_times=None,
                 current_federal_funds_rate=None,
                 current_nikkei_price=None,
                 correlation_foreign_asset_and_domestic_risk_free_rate=None,
                 correlation_foreign_asset_and_exchange_rate=None,
                 foreign_risk_free_rate=None,
                 nikkei_dividends=None,
                 foreign_asset_vol=None,
                 exchange_rate_vol=None,
                 random_seed=None,
                 ):
        self.bond_data = bond_data
        self.domestic_risk_free_rate_vol = domestic_risk_free_rate_vol
        self.mean_reversion_speed = mean_reversion_speed
        self.maturity_T = maturity_T
        self.iteration_times = iteration_times
        self.timestep = maturity_T / iteration_times
        self.timeline = [self.timestep * i for i in range(iteration_times+1)]
        self.domestic_risk_free_rate_list = [0] * (iteration_times + 1)
        self.nikkei_price_list = [0] * (iteration_times + 1)
        self.current_federal_funds_rate = current_federal_funds_rate
        self.current_nikkei_price = current_nikkei_price
        self.correlation_foreign_asset_and_domestic_risk_free_rate = correlation_foreign_asset_and_domestic_risk_free_rate
        self.correlation_foreign_asset_and_exchange_rate = correlation_foreign_asset_and_exchange_rate
        self.foreign_risk_free_rate = foreign_risk_free_rate
        self.nikkei_dividends = nikkei_dividends
        self.foreign_asset_vol = foreign_asset_vol
        self.exchange_rate_vol = exchange_rate_vol
        self.Delta = Delta
        self.strike_price_nikkei = strike_price_nikkei
        self.strike_price_libor = strike_price_libor
        self.random_seed = random_seed

    def start_monte_carlo(self):
        """
        Conduct Monte Carlo Simulation for a single time
        """
        # Initialize two lists with current data
        self.domestic_risk_free_rate_list[0] = self.current_federal_funds_rate
        self.nikkei_price_list[0] = self.current_nikkei_price

        np.random.seed(self.random_seed)
        rands = np.random.normal(0, 1, [self.iteration_times, 2])  # (mean, std, size)

        for i in range(self.iteration_times):
            # Initialize two independent standard normal random variables for each iteration
            epsilon1 = rands[i,0]  # standard normal distribution
            epsilon2 = rands[i,1]  # another independent standard normal distribution

            # recursive relation for the domestic risk-free rate
            self.domestic_risk_free_rate_list[i + 1] = self.domestic_risk_free_rate_list[i] + \
                                                       (
                                                            self.Theta(i * self.timestep)
                                                            - self.mean_reversion_speed * self.domestic_risk_free_rate_list[i]
                                                       ) * self.timestep \
                                                       + self.domestic_risk_free_rate_vol * np.sqrt(self.timestep) * \
                                                       ( # Chelosky Factorization
                                                            self.correlation_foreign_asset_and_domestic_risk_free_rate * epsilon2
                                                            + np.sqrt(1 - self.correlation_foreign_asset_and_domestic_risk_free_rate ** 2) * epsilon1
                                                       )
            # recursive relation for the nikkei price
            self.nikkei_price_list[i + 1] = self.nikkei_price_list[i] + \
                                            self.nikkei_price_list[i] * \
                                            (
                                                self.foreign_risk_free_rate
                                                - self.nikkei_dividends
                                                - self.correlation_foreign_asset_and_exchange_rate * self.foreign_asset_vol * self.exchange_rate_vol
                                            ) * self.timestep \
                                            + self.nikkei_price_list[i] * self.foreign_asset_vol * np.sqrt(self.timestep) * epsilon2

    def get_domestic_risk_free_rate_at_a_specified_time(self, time):  # time in years
        """
        Get domestic risk-free rate simulated by Monte Carlo at a specific time
        :param time: time in years (< than maturity_T)
        :return: simulated domestic risk-free rate (annualized)
        """
        index_in_list = int(time / self.timestep)
        return self.domestic_risk_free_rate_list[index_in_list]

    def get_nikkei_price_at_a_specified_time(self, time):  # time in years
        """
        Get Nikkei Price simulated by Monte Carlo at a specific time
        :param time: time in years (< than maturity_T)
        :return: simulated Nikkei Price
        """
        index_in_list = int(time / self.timestep)
        return self.nikkei_price_list[index_in_list]

    def libor_rate_ratio(self, t, T):
        """
        This is the libor ratio defined as: L(t; t, T)/L(0; t, T) = (1/p(t,T) - 1) * p(0,T) / (p(0,t)-p(0,T))
        :param t: time in years
        :param T: time in years
        :return: libor rate ratio
        """
        return self.bond_data.get_value_on_cubic_spline_for_bond_prices(T) / \
               (
                       self.bond_data.get_value_on_cubic_spline_for_bond_prices(t)
                       - self.bond_data.get_value_on_cubic_spline_for_bond_prices(T)
               ) \
               * (1/self.p_t_T(t, T, self.get_domestic_risk_free_rate_at_a_specified_time(t)) - 1)

    def price_Exotic_Option_at_maturity(self):
        """
        Calculate the price of Exotic Option at maturity_T
        :return: The price of Exotic Option at maturity_T
        """
        return max(
                    0,
                    (self.get_nikkei_price_at_a_specified_time(self.maturity_T) / self.current_nikkei_price - self.strike_price_nikkei)
                    *
                    (self.strike_price_libor - self.libor_rate_ratio(self.maturity_T - self.Delta, self.maturity_T))
                  )

    def discount_factor(self, T):
        """
        Calculate the discount factor for maturity_T: -\int_0^T r_s ds
        :param T: time in years
        :return: the value of the discount factor
        """
        sum_of_short_rates = 0
        end_index = int(T / self.timestep)
        for i in range(end_index):
            sum_of_short_rates += self.domestic_risk_free_rate_list[i] * self.timestep
        return np.exp(-sum_of_short_rates)

    def current_price_Exotic_Option(self):
        """
        Calculate the price of Exotic Option at current time
        :return: the value of the price of Exotic Option at current time
        """
        return self.discount_factor(self.maturity_T) * self.price_Exotic_Option_at_maturity()

    def G(self, t):
        """
        The function is defined as: G(t) = 0.5 * r_d^2 * B(0,t)^2
        :param t: time in years
        :return: the value of G(t)
        """
        return 0.5 * self.domestic_risk_free_rate_vol ** 2 * self.B_t_T(0, t) ** 2

    def first_derivative_of_G(self, t):
        """
        The function is defined as: dG(t)/dt = r_d^2 * B(0,t) * exp(-a * t)
        :param t: time in years
        :return: the value of dG(t)/dt
        """
        return self.domestic_risk_free_rate_vol ** 2 * self.B_t_T(0, t) * np.exp(-self.mean_reversion_speed * t)

    def Theta(self, t):
        """
        The function is defined as: Theta(t) = f_T(0,t) + dG(t)/dt + a * (f(0,t) + G(t))
        :param t: time in years
        :return: the value of Theta(t)
        """
        return self.bond_data.get_value_for_T_derivative_of_forward_rate(t) + self.first_derivative_of_G(t) \
               + self.mean_reversion_speed * (self.bond_data.get_value_for_forward_rate(t) + self.G(t))

    def B_t_T(self, t, T):
        """
        The function is defined as: B(t, T) = 1/a * (1 - exp(-a * (T-t)))
        :param t: time in years
        :param T: time in years
        :return: the value of B(t, T)
        """
        return 1 / self.mean_reversion_speed * (1 - np.exp(-self.mean_reversion_speed * (T - t)))

    def p_t_T(self, t, T, r_t):
        """
        The function is defined as:
        p(t, T) = p(0, T) / p(0, t) * exp(B(t, T) * f(0,t) - 1/(4a) * sigma_d^2 * B^2(t, T) * (1 - exp(-2 * a * t)) - B(t, T) * r_t)
        :param t: time in years
        :param T: time in years
        :param r_t: simulated domestic risk-free rate at time t
        :return: the value of function p(t, T)
        """
        return self.bond_data.get_value_on_cubic_spline_for_bond_prices(T) / self.bond_data.get_value_on_cubic_spline_for_bond_prices(t) \
               * np.exp(
                   self.B_t_T(t, T) * self.bond_data.get_value_for_forward_rate(t)
                   - 1 / (4 * self.mean_reversion_speed) * self.domestic_risk_free_rate_vol ** 2 * self.B_t_T(t, T) ** 2 * (1 - np.exp(-2 * self.mean_reversion_speed * t))
                   - self.B_t_T(t, T) * r_t
               )
