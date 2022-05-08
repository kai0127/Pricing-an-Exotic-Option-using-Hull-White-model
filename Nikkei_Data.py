#  Copyright (c) 2022/5/5 下午5:41 Last Modified By Kaiwen Zhou

"""
This module/class contains one method by which we can find data from https://indexes.nikkei.co.jp such as,
the latest dividend yield for Nikkei 225
i.e. q^f

@created 05/05/2022 - 5:41 PM
@author Kaiwen Zhou
"""
import requests
from bs4 import BeautifulSoup


class Nikkei_Data(object):

    def __init__(self):
        self.url = 'https://indexes.nikkei.co.jp/en/nkave/archives/data?list=dividend'

    def get_latest_dividend_yield(self):
        """
        Get the latest dividend yield via BeautifulSoup from the provided url
        :return: the value of the latest dividend yield (annualized & not in percentage)
        """
        req = requests.get(self.url)
        soup = BeautifulSoup(req.content, 'html.parser')

        # get the date where the latest data for Simple Average(%) of dividends yields is on
        last_update_date = soup.find('div', class_="last-update-cal").getText().replace('Update：', '').strip()

        # find all html Table Data Cell element tagged with <td>
        list_of_html_table_data_cells = soup.find_all('td')

        # this for loop is to find the date in the table which is the current date
        pivot = 0  # index of the list_of_html_table_data_cells at which its date (its content) is the current date
        for i in range(len(list_of_html_table_data_cells)):
            if list_of_html_table_data_cells[i].getText() == last_update_date:
                pivot = i
                break
        # The dividend yields corresponding to this date is one cell after the
        # pivot cell, (i.e. list_of_html_table_data_cells[pivot]).
        return float(list_of_html_table_data_cells[pivot+1].getText()) / 100  # convert percentage to decimal


