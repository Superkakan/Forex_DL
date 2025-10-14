# Made by Felix Stockinger and Lukas Ydkvist
# Forex DL

A DL project to predict currency prices using a LSTM and semantic data.
 

1m Data downloaded from:
https://www.histdata.com/ 
Details about 1m data:
https://www.histdata.com/f-a-q/data-files-detailed-specification/

1h/1d data from Yahoo Finance using yfinance library

Semantic Data from ecb and federal reserve site:
https://www.federalreserve.gov/newsevents/pressreleases.htm
https://www.ecb.europa.eu/home/html/index.en.html


As expected the future prediction doesnt really work.
Next step can be to include intraday prices of currencies or economic indicators, for a more accurate prediction.
The impact of the semantic data is minimal, could research on how to make this weight in more.
The tuning/testing of hyperparameter can also be automated

Dependencies:
yfinance
pandas
sklearn
