from alpha_vantage_try.timeseries import TimeSeries
import pprint
# ts = TimeSeries(key='2FBDH872GJN0OA8B', output_format='pandas')
# data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')
#
# pprint(data.head(2))

from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='2FBDH872GJN0OA8B')
# Get json object with the intraday data and another with  the call's metadata
data, meta_data = ts.get_intraday('GOOGL')

