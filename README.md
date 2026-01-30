# Made by Felix Stockinger and Lukas Ydkvist

A DL project to predict currency prices using a LSTM and semantic data. 
 

**1m Data downloaded from:** <br />
https://www.histdata.com/ <br /> <br />
**Details about 1m data:** <br />
https://www.histdata.com/f-a-q/data-files-detailed-specification/ <br />

**1h/1d data from Yahoo Finance using yfinance library.**

**Semantic Data from ecb and federal reserve site:** <br />
https://www.federalreserve.gov/newsevents/pressreleases.htm <br />
https://www.ecb.europa.eu/home/html/index.en.html

**Dependencies:**
yfinance
pandas
sklearn

Anyway, next step can be to include intraday prices of currencies or economic indicators, for a more accurate prediction.<br />
The impact of the semantic data is minimal, could research on how to make this weight in more.<br />
The tuning/testing of hyperparameter can also be automated


## Some graphs:


### 1m, 10 epochs, 0.001 LR
<img width="2560" height="1335" alt="1m 10 Epochs 0 001 LR" src="https://github.com/user-attachments/assets/faaa1384-bd90-4f28-a6c2-c17c8412831f" />

### 1m, 10 epochs, 0.001 LR, Future Prediction
<img width="2560" height="1335" alt="1m 10 Epochs 0 001 LR Future" src="https://github.com/user-attachments/assets/5325e52f-2bbc-4820-a002-914e42f2a0a2" />
