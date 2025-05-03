python3 main.py --paper

python3 main.py --download --symbols AAPL,TSLA --interval 5m --period 30d --indicators --format csv

python3 main.py --backtest SimpleStrategy --start-date 2025-04-20 --end-date 2025-05-01 --symbols TSLA --initial-capital 100000