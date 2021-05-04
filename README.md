Pairs_trading aim to help the algorithmic trading community to find the potential candidates base on metrics like cointegrating relationship between the candidates, stationarity of the spreads, level of persistent of the series, the frequency of mean reversion and also the half-life of the spreads for pairs trading.

1) add(syms, start, end)
   add symbols and desired time period 

2) bring_in(price_df, start, end, type='candidates')
   import dataframe instead of adding a new one

3) restore():
   using the backup dataframe as a new one

4) remove(syms)
   removing symbols 

5) find_candidates()
   search candidates for pair trading

6) plot(index):
   visualize the results 
