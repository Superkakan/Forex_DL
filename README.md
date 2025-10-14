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

Although the model correctly predicts an upgoing trend in one of the predict the future graphs, it doesnt really work, because its a LSTM. But who knows what will happen with more work?<br /><br />
Anyway, next step can be to include intraday prices of currencies or economic indicators, for a more accurate prediction.<br />
The impact of the semantic data is minimal, could research on how to make this weight in more.<br />
The tuning/testing of hyperparameter can also be automated


All the saved graphs that doesnt have the timeframe in the name, have a granuality of 1h.

## Some graphs:

### 1d, 50 epochs, 0.01 LR, 24 days future prediction
<img width="1920" height="975" alt="1d 50e 0 01 LR" src="https://github.com/user-attachments/assets/f27555bc-b12a-43d5-94db-f5f229d8d0cc" />

### 1d, 50 epochs, 0.01 LR
<img width="1920" height="975" alt="1d 50e 0 01 LR 24days" src="https://github.com/user-attachments/assets/7be5f08d-074a-49f5-9263-908808461ea6" />

### 1d, 340 epochs, 0.001 LR 
<img width="1920" height="975" alt="1d 340 epochs 0 001 LR" src="https://github.com/user-attachments/assets/3d4180b6-d637-4b51-a141-0556bb04f760" />

### 1d, 340 epochs, 0.001 LR, 24 days future prediction
<img width="1920" height="975" alt="1d 340e 0 001 LR 24days" src="https://github.com/user-attachments/assets/b95dfa97-02bd-45ca-8778-8eea462c0f66" />
