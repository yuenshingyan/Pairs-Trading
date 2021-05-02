#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import pmdarima as pm
from datetime import date
from dateutil.relativedelta import relativedelta
import yfinance as yf
import statsmodels.tsa.stattools as ts
import datetime
import itertools
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import sys

class pairs_trading:
    def __init__(self):
        self.stocks = {}
          
    def add(self, syms, start, end):
        if type(syms) == pd.Series:
            syms = list(syms)
            syms = [s.replace('^', '-') for s in syms]
            
        if type(syms) == list or type(syms) == tuple:
            syms = [s.replace('^', '-') for s in syms]
            for sym in syms:
                try:
                    self.stocks[f'{sym}'] = yf.download(sym, threads= False, start=start, end=end).dropna()
                except:
                    continue
                    
        if type(syms) == str:
            syms = syms.replace('^', '-')
            self.stocks[f'{syms}'] = yf.download(syms, start=start, end=end).dropna()
        
        for sym in list(self.stocks.keys()):
            if self.stocks[sym].shape[0] < 251:
                del self.stocks[sym]
                
        df2 = pd.concat(self.stocks)
        self.prices = pd.pivot_table(data = df2, values='Close', index='Date', columns=df2.index.get_level_values(0))
        self.prices_back_up = self.prices.copy()
        
    def bring_in(self, price_df, start, end, type='candidates'):
        if type == 'candidates':
            self.candidates = price_df
            
        can = []
        for i in range(len(self.candidates['candidate'])):
            for j in range(2):
                can.append(self.candidates['candidate'][i][j])
        can = set(can)        
        
        self.syms = []
        for c in can:
            try:
                self.stocks[f'{c}'] = yf.download(c, threads= False, start=start, end=end).dropna()
                self.syms.append(c)
            except:
                continue
                
        df2 = pd.concat(self.stocks)
        self.prices = pd.pivot_table(data = df2, values='Close', index='Date', columns=df2.index.get_level_values(0))
        self.prices_back_up = self.prices.copy()

        if type == 'prices':
            self.prices = price_df
    
    def restore(self):
        self.prices = self.prices_back_up
    
    def remove(self, syms):
        if type(syms) == pd.Series:
            syms = list(syms)
            syms = [s.replace('^', '-') for s in syms]
            
        if type(syms) == str:
            del self.stocks[syms]
        if type(syms) == list or type(sym) == tuple:
            for sym in syms:
                del self.stocks[sym]  
    
    def find_candidates(self):
        df=self.prices.dropna(axis=1)
        # https://stackoverflow.com/questions/59399889/understanding-the-generalized-formula-of-the-hurst-exponent-in-python
        def HurstEXP( ts = [ None, ] ): # degree of mean-reversion
            #---------------------------------------------------------------------------------------------------------------------------<self-reflective>
            if ( ts[0] == None ): # DEMO: Create a SYNTH Geometric Brownian Motion, Mean-Reverting and Trending Series:
                gbm = np.log( 1000 + np.cumsum(     np.random.randn( 100000 ) ) )  # a Geometric Brownian Motion[log(1000 + rand), log(1000 + rand + rand ), log(1000 + rand + rand + rand ),... log(  1000 + rand + ... )]
                mr  = np.log( 1000 +                np.random.randn( 100000 )   )  # a Mean-Reverting Series    [log(1000 + rand), log(1000 + rand        ), log(1000 + rand               ),... log(  1000 + rand       )]
                tr  = np.log( 1000 + np.cumsum( 1 + np.random.randn( 100000 ) ) )  # a Trending Series          [log(1001 + rand), log(1002 + rand + rand ), log(1003 + rand + rand + rand ),... log(101000 + rand + ... )]

            # Output the Hurst Exponent for each of the above SYNTH series
                print ( "HurstEXP( Geometric Browian Motion ):   {0: > 12.8f}".format( HurstEXP( gbm ) ) )
                print ( "HurstEXP(    Mean-Reverting Series ):   {0: > 12.8f}".format( HurstEXP( mr  ) ) )
                print ( "HurstEXP(          Trending Series ):   {0: > 12.8f}".format( HurstEXP( tr  ) ) )

                return ( "SYNTH series demo ( on HurstEXP( ts == [ None, ] ) ) # actual numbers vary, as per np.random.randn() generator" )

            pass;     too_short_list = 101 - len( ts )                  # MUST HAVE 101+ ELEMENTS
            if ( 0 <  too_short_list ):                                 # IF NOT:
                 ts = too_short_list * ts[:1].values + ts.values                      #    PRE-PEND SUFFICIENT NUMBER of [ts[0],]-as-list REPLICAS TO THE LIST-HEAD
            #---------------------------------------------------------------------------------------------------------------------------
            lags = range( 2, 100 )                                                              # Create the range of lag values
            tau  = [ np.sqrt( np.std( np.subtract( ts[lag:].values, ts[:-lag].values ) ) ) for lag in lags ]  # Calculate the array of the variances of the lagged differences

            #oly = np.polyfit( np.log( lags ), np.log( tau ), 1 )                               # Use a linear fit to estimate the Hurst Exponent
            #eturn ( 2.0 * poly[0] )                                                            # Return the Hurst exponent from the polyfit output
            H = ( 2.0 * np.polyfit( np.log( lags ), np.log( tau ), 1 )[0] )

            return H
        
        def halflife(z):
            #set up lagged series of z_array and return series of z_array
            z_lag = np.roll(z,1)
            z_lag[0] = 0
            z_ret = z - z_lag
            z_ret[0] = 0

            #run OLS regression to find regression coefficient to use as "theta"
            model = sm.OLS(z_ret, z_lag, window_type='rolling', window=30)
            res = model.fit()

            #calculate halflife
            halflife = -log(2) / res.params[0]
            return halflife
        
        def count_peaks(tick1, tick2):
            def zscore(series):
                return (series - series.mean()) / np.std(series)
            # Calculating the spread
            S1 = df[tick1]
            S2 = df[tick2]

            S1 = sm.add_constant(S1)
            results = sm.OLS(S2, S1).fit()
            S1 = S1[tick1]
            b = results.params[tick1]

            spread = S2 - b * S1

            ratio = S1/S2
            yhat = savgol_filter(zscore(ratio), 5, 3)
            peaks, _ = find_peaks(yhat, height=1, distance=30)
            peaks2, _ = find_peaks(-yhat, height=1, distance=30)
            return len(peaks)+len(peaks2)

        # find co integrated paris----------------------------------------------------------------------------------------------
        total = int(len(df.columns) * (len(df.columns) - 1) / 2)
        count = 1
        p_coin = []
        stock1 = []
        stock2 = []
        
        print('Finding cointegrated pairs: ')
        
        for com in itertools.combinations(df.columns, 2):
            try:
                result = coint(df[com[0]], df[com[1]])
                sys.stdout.write(f'    {count}/{total} {com[0]}, {com[1]}\r')
                sys.stdout.flush()
                count = count + 1
                if result[1] < 0.05:
                    p_coin.append(result[1])
                    stock1.append(com[0])
                    stock2.append(com[1])
            except:
                print(f'    Cannot perform cointegration test with {com}\r')
                total = total - 1
                
        pairs = pd.DataFrame([stock1, stock2, p_coin], index=['stock1', 'stock2', 'p']).T
        pairs = pairs.sort_values('p')

        print(' ')
        print(f'    {len(pairs)}/{total} are cointegrated.')
        print(' ')

        count = 1
        stationary = []
        mean = []
        std = []
        zscore = []
        p_adfuller = []
        halflife_list = []
        p_adfuller2 = []
        H_list = []
        peaks = []
        print('Finding stationary spreads: ')
        for st1, st2 in zip(pairs['stock1'], pairs['stock2']):
            S1 = df[st1]
            S2 = df[st2]
            model = sm.OLS(endog=S1, exog=S2, window_type='rolling', window=30).fit()
            hedgeRatio = model.params[0]
            z = S1 - hedgeRatio * S2

            p_adfuller = ts.adfuller(z)[1]
            H = HurstEXP(z)
            sys.stdout.write(f'    {count}/{len(pairs)} {st1}, {st2}\r')
            sys.stdout.flush()
            count = count + 1
            no_peaks = count_peaks(st1, st2)
            if p_adfuller < 0.05 and H < 0.5 and halflife(z) > 1 and halflife(z) < 365:
                stationary.append((st1, st2))
                p_adfuller2.append(p_adfuller)
                halflife_list.append(halflife(z))
                H_list.append(H)
                peaks.append(no_peaks)

        candidate = pd.DataFrame([stationary, p_adfuller2, halflife_list, peaks, H_list], index=['candidate', 'p_adfuller', 'halflife', 'peaks', 'HurstEXP']).T
        candidate = pd.DataFrame(np.hstack([candidate, pd.DataFrame(pairs['p'][0:len(candidate)])]), columns=['candidate', 'p_adfuller', 'halflife', 'peaks', 'HurstEXP', 'p_coin'])

        candidate = candidate.sort_values('p_adfuller')
        
        print(' ')
        print(f'    {len(candidate)}/{len(pairs)} match the criteria.')
        print(' ')
        
        if candidate.shape[0] == 0:
            print(' ')
            print('    No ideal pairs found.')
        else:    
            self.candidates = candidate[candidate['peaks'] > 11 ]
        
        print(f'    {len(self.candidates)} Candidates have peaks more than 11.') 
        
        if len(self.candidates) == 0:
            print(' ')
            print('    No ideal pairs found.')
        
        if len(self.candidates) > 0:
        
            def zscore(series):
                return (series - series.mean()) / np.std(series)

            profitability = []
            for i in range(len(self.candidates)):
                tick1, tick2 = self.candidates.iloc[i][0][0], self.candidates.iloc[i][0][1]
                # Calculating the spread
                S1 = self.prices[tick1]
                S2 = self.prices[tick2]

                S1 = sm.add_constant(S1)
                results = sm.OLS(S2, S1).fit()
                S1 = S1[tick1]
                b = results.params[tick1]

                spread = S2 - b * S1

                yhat = savgol_filter(zscore(spread), 5, 3)
                peaks, _ = find_peaks(yhat, height=2, distance=30)
                bottoms, _ = find_peaks(-yhat, height=2, distance=30)

                profitability.append(len(peaks) + len(bottoms))

            self.candidates = pd.DataFrame(np.hstack([self.candidates, pd.DataFrame(profitability)]), 
            columns=['candidate', 'p_adfuller', 'halflife', 'peaks', 'HurstEXP', 'p_coin', 'profitability']).sort_values('profitability', ascending=False)
            self.candidates = self.candidates[self.candidates['profitability'] > 0 ]
            self.candidates.index = pd.Index([idx for idx in range(len(self.candidates))])

            print(f'    {len(self.candidates)} are profitable.')
        
    def plot(self, index):
        def zscore(series):
            return (series - series.mean()) / np.std(series)
        
        tick1, tick2 = self.candidates['candidate'][index][0], self.candidates['candidate'][index][1]
        # Calculating the spread
        S1 = self.prices[tick1]
        S2 = self.prices[tick2]

        S1 = sm.add_constant(S1)
        results = sm.OLS(S2, S1).fit()
        S1 = S1[tick1]
        b = results.params[tick1]

        spread = S2 - b * S1

        ratio = S1/S2
        yhat = savgol_filter(zscore(spread), 5, 3)
        peaks, _ = find_peaks(yhat, height=2, distance=30)
        bottoms, _ = find_peaks(-yhat, height=2, distance=30)
        zero_crossings = np.where(np.diff(np.sign(yhat)))[0]
        exit1 = np.where((zscore(spread) >= -2) & (zscore(spread) < -1.5), yhat, np.nan)
        exit2 = np.where((zscore(spread) <= 2) & (zscore(spread) > 1.5), yhat, np.nan)
        spread.name = 'spread'      
        
        cloest_exits = []
        for i in range(len(peaks)):
            try:
                cloest_exits.append(np.where(zscore(spread)[bottoms].index[i] - zscore(spread)[zero_crossings].index < datetime.timedelta(0))[0][0])
            except:
                continue
        
        fig, axs = plt.subplots(2, figsize=(15,15))

        axs[0].plot(zscore(spread)[peaks].index, yhat[peaks], 'ro', color='red')
        axs[0].plot(zscore(spread)[bottoms].index, yhat[bottoms], 'ro', color='green')
        axs[0].plot(zscore(spread).index, exit1, 'ro', color='red', alpha=0.5)
        axs[0].plot(zscore(spread).index, exit2, 'ro', color='green', alpha=0.5)
        #plt.plot(zscore(spread)[zero_crossings].index[cloest_exits], yhat[zero_crossings][cloest_exits], 'ro', color='orange')
        axs[0].plot(zscore(spread).index, yhat)
        axs[0].plot(zscore(ratio).index, zscore(ratio), alpha=0.25)
        
        axs[0].axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
        axs[0].axhline(1, color='red', linestyle=':', alpha=0.5, linewidth=0.5)
        axs[0].axhline(-1, color='green', linestyle=':', alpha=0.5, linewidth=0.5)
        axs[0].axhline(2, color='red', linestyle='--', alpha=0.75, linewidth=0.5)
        axs[0].axhline(-2, color='green', linestyle='--', alpha=0.75, linewidth=0.5)
        axs[0].axhline(3, color='red', alpha=1, linewidth=0.5)
        axs[0].axhline(-3, color='green', alpha=1, linewidth=0.5)
        axs[0].set(ylabel='Z Score', title=str(tick1) + '-' + str(tick2) + ' Spread')
        axs[0].legend([f'Short {tick2}, Long {tick1}', f'Long {tick2}, Short {tick1}', 'Exit (Sell both)', 'Z Score (Spread)', 'Z Score (Beta)'],
                   bbox_to_anchor=(1.2, 1))
        
        axs[1].plot(S1.index, S1)
        axs[1].plot(S2.index, S2)
        axs[1].set(ylabel='Price (USD)')
        axs[1].legend([tick1, tick2],
                   bbox_to_anchor=(1.2, 1))
        
        print(len(yhat[peaks]) + len(yhat[bottoms]), f' trading opportunities identified from {str(zscore(ratio).index[0])[0:10]} to {str(zscore(ratio).index[-1])[0:10]} for {tick1} - {tick2} Spread')
        
        if len(yhat[bottoms]) > 0 and len(yhat[peaks]) > 0:
            return pd.concat([S1[zscore(spread)[peaks].index], S1[zscore(spread)[bottoms].index]], axis=0).sort_index(), pd.concat([S2[zscore(spread)[peaks].index], S2[zscore(spread)[bottoms].index]], axis=0).sort_index(), S1[zscore(spread)[zero_crossings].index[cloest_exits]].sort_index(), S2[zscore(spread)[zero_crossings].index[cloest_exits]].sort_index()
        
        if len(yhat[bottoms]) > 0 and len(yhat[peaks]) == 0:
            return S1[zscore(spread)[bottoms].index], S2[zscore(spread)[bottoms].index], S1[zscore(spread)[zero_crossings].index], S2[zscore(spread)[zero_crossings].index]
        
        if len(yhat[bottoms]) == 0 and len(yhat[peaks]) > 0:
            return S1[zscore(spread)[peaks].index], S2[zscore(spread)[peaks].index], S1[zscore(spread)[zero_crossings].index], S2[zscore(spread)[zero_crossings].index]
