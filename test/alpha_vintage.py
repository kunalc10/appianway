from polygon import RESTClient


# def main():
#     key = "dI1eO3CPUBmowtqS0XBm6vFXWSHi4tzR"
#
#     # RESTClient can be used as a context manager to facilitate closing the underlying http session
#     # https://requests.readthedocs.io/en/master/user/advanced/#session-objects
#     with RESTClient(key) as client:
#         resp = client.stocks_equities_daily_open_close("AAPL", "2021-06-11")
#         print(f"On: {resp.from_} Apple opened at {resp.open} and closed at {resp.close}")
#
#
# if __name__ == '__main__':
#     main()


import datetime

from polygon import RESTClient


def ts_to_datetime(ts) -> str:
    return datetime.datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M')


def main():
    key = "2FBDH872GJN0OA8B"

    # RESTClient can be used as a context manager to facilitate closing the underlying http session
    # https://requests.readthedocs.io/en/master/user/advanced/#session-objects
    with RESTClient(key) as client:
        from_ = "2019-01-01"
        to = "2019-02-01"
        resp = client.stocks_equities_aggregates("AAPL", 1, "day", from_, to, unadjusted=False)

        print(f"Minute aggregates for {resp.ticker} between {from_} and {to}.")

        for result in resp.results:
            dt = ts_to_datetime(result["t"])
            print(f"{dt}\n\tO: {result['o']}\n\tH: {result['h']}\n\tL: {result['l']}\n\tC: {result['c']} ")


if __name__ == '__main__':
    main()