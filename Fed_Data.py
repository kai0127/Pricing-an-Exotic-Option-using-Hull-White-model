#  Copyright (c) 2022/5/7 下午11:28 Last Modified By Kaiwen Zhou

"""
This module/class contains methods and functions by which we can find data from the FRED such as,
current federal funds rate, the latest average Nikkei 225 price, and
the correlation between the Nikkei prices and USD/JPY exchange spot rates during the past 180 days
i.e. r_d(t=0), S^f(0), rho_sx

@created 05/04/2022 - 2:24 PM
@author Kaiwen Zhou
"""
import math
import fredapi
import datetime
import numpy as np


class Fed_Data(object):

    def __init__(self, api_key):
        self.fred = fredapi.Fred(api_key=api_key)

    def get_current_federal_funds_rate(self):
        """
        This method is for us to extract the federal funds rate via the fredapi
        :return: the federal funds rate for the current month, since the Fed changes it in a monthly basis (annualized)
        """
        # Get the federal funds rate for the current month or the month before, since the Fed might not have
        # decided the federal funds rate for the current month
        first_day_of_this_month = str(datetime.date.today().replace(day=1))
        last_month = datetime.date.today().month - 1
        last_month_same_day = str(datetime.date.today().replace(month=last_month))

        # Extract and return only the value in the dataframe
        return self.fred.get_series('FEDFUNDS', last_month_same_day, first_day_of_this_month).item() / 100

    def get_latest_daily_average_nikkei_price(self):
        # Get the latest data for average Nikkei 225 price for a 7 days period
        # (the FRED updates the data daily but not on weekends or holidays)
        today_date = str(datetime.date.today())
        date_seven_days_before = str(datetime.date.today() - datetime.timedelta(hours=24 * 7))

        # Extract and return only the value of the latest data in the dataframe
        return self.fred.get_series('NIKKEI225', date_seven_days_before, today_date).iloc[-1]

    def get_correlation_foreign_asset_and_exchange_rate(self):  # Using historical data during the past 180 days
        """
        By Cholesky Factorization, we only need to find out the correlation between two Wiener Processes dW_S, dW_X
        which turns out to be exactly the correlation between processes {S(i+1)- S(i) / S(i)} and {X(i+1)- X(i) / X(i)}
        where S(i) = the Nikkei price at t_i = i * delta_t; X(i) = the USD/JPY exchange spot rate at t_i = i * delta_t;
        dW_S is the Wiener process for Nikkei price process (GBM);
        dW_X is the Wiener process for USD/JPY exchange spot rate process (GBM)

        We utilize this method to find the correlation between processes {S(i+1)- S(i) / S(i)} and {X(i+1)- X(i) / X(i)}
        using data during the past 180 days for each process extracted via fredapi
        :return: the correlation between the Nikkei prices and USD/JPY exchange spot rates during the past 180 days
        """
        # Get the latest data for average Nikkei 225 price for a 180 days period
        # (the FRED updates the data daily but not on weekends or holidays)
        today_date = str(datetime.date.today())
        date_180_days_before = str(datetime.date.today() - datetime.timedelta(hours=24 * 180))

        # Get raw data for both processes via fredapi
        df_nikkei_prices = self.fred.get_series('NIKKEI225', date_180_days_before, today_date)
        df_jpy_usd = self.fred.get_series('DEXJPUS', date_180_days_before, today_date)

        # Trim out the 'NaN' value in the raw data for both processes
        list_nikkei_prices = [df_nikkei_prices.iloc[i] for i in range(len(df_nikkei_prices)) if not math.isnan(df_nikkei_prices.iloc[i])]
        # We have to take reciprocal here, because in our assumption X_t is USD/JPY
        list_usd_jpy = [1/df_jpy_usd.iloc[i] for i in range(len(df_jpy_usd)) if not math.isnan(df_jpy_usd.iloc[i])]

        # Adjust two series of data to have the same length
        minimum_length = min(len(list_nikkei_prices), len(list_usd_jpy))
        adjusted_list_nikkei_prices = list_nikkei_prices[len(list_nikkei_prices)-minimum_length:len(list_nikkei_prices)]
        adjusted_list_usd_jpy = list_usd_jpy[len(list_usd_jpy)-minimum_length:len(list_usd_jpy)]

        # Compute processes {S(i+1)- S(i) / S(i)} and {X(i+1)- X(i) / X(i)}
        target_process_nikkei = [(adjusted_list_nikkei_prices[i+1]-adjusted_list_nikkei_prices[i])/adjusted_list_nikkei_prices[i] for i in range(minimum_length-1)]
        target_process_usd_jpy = [(adjusted_list_usd_jpy[i+1]-adjusted_list_usd_jpy[i])/adjusted_list_usd_jpy[i] for i in range(minimum_length-1)]

        return np.corrcoef(target_process_nikkei, target_process_usd_jpy)[0, 1]  # the correlation
