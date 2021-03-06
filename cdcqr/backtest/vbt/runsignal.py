#ct.alfafactory

import logging
import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"

import vectorbt as vbt
import backtrader as bt
from numba import njit

import numpy as np
import pandas as pd
import inspect 

DF = pd.DataFrame

class Sizer99Percent(bt.sizers.PercentSizer):
    params = (('percents', 99),)

class Sizer10Percent(bt.sizers.PercentSizer):
    params = (('percents', 10),)   

def printTradeAnalysis(cerebro, analyzers,icap=10000):    
    def pretty_print(format, *args): #can print into string/file
        print(format.format(*args))
    def exists(object, *properties):
        for property in properties:
            if not property in object: return False
            object = object.get(property)
        return True
    format = "  {} : {}"
    NA     = '-'
    print('Backtesting Results')
    if hasattr(analyzers, 'ta'):
        ta = analyzers.ta.get_analysis()
        openTotal         = ta.total.open          if exists(ta, 'total', 'open'  ) else None
        closedTotal       = ta.total.closed        if exists(ta, 'total', 'closed') else None
        pnlNetTotal       = ta.pnl.net.total       if exists(ta, 'pnl', 'net', 'total'  ) else None
        pnlNetAverage     = ta.pnl.net.average     if exists(ta, 'pnl', 'net', 'average') else None
        pretty_print(format, 'Open Positions', openTotal   or NA)
        pretty_print(format, 'Closed Trades',  closedTotal or NA)
        pretty_print(format, 'Inital Portfolio Value', '${}'.format(icap))
        pretty_print(format, 'Final Portfolio Value',  '${}'.format(cerebro.broker.getvalue()))
        pretty_print(format, 'Net P/L',                '${}'.format(round(pnlNetTotal,   2)) if pnlNetTotal   else NA)
        pretty_print(format, 'P/L Average per trade',  '${}'.format(round(pnlNetAverage, 2)) if pnlNetAverage else NA)
    if hasattr(analyzers, 'drawdown'):
        pretty_print(format, 'Drawdown', '${}'.format(analyzers.drawdown.get_analysis()['drawdown']))
    if hasattr(analyzers, 'sharpe'):
        pretty_print(format, 'Sharpe Ratio:', analyzers.sharpe.get_analysis()['sharperatio'])



class trade_list(bt.Analyzer):
    '''
    Records closed trades and returns dictionary containing the following
    keys/values:
      - ``ref``: reference number (from backtrader)
      - ``ticker``: data name
      - ``dir``: direction (long or short)
      - ``datein``: entry date/time
      - ``pricein``: entry price (considering multiple entries)
      - ``dateout``: exit date/time
      - ``priceout``: exit price (considering multiple exits)
      - ``chng%``: price change in %s during trade
      - ``pnl``: profit/loss
      - ``pnl%``: profit/loss in % to broker value
      - ``size``: size
      - ``value``: value
      - ``cumpnl``: cumulative profit/loss for trades shown before this trade
      - ``nbars``: average trade duration in price bars
      - ``pnl/bar``: average profit/loss per bar
      - ``mfe``: max favorable excursion in $s from entry price
      - ``mae``: max adverse excursion in $s from entry price
      - ``mfe%``: max favorable excursion in % of entry price
      - ``mae%``: max adverse excursion in % of entry price
    '''

    def get_analysis(self):

        return self.trades


    def __init__(self):

        self.trades = []
        self.cumprofit = 0.0


    def notify_trade(self, trade):

        if trade.isclosed:

            brokervalue = self.strategy.broker.getvalue()

            dir = 'short'
            if trade.history[0].event.size > 0: dir = 'long'

            pricein = trade.history[len(trade.history)-1].status.price
            priceout = trade.history[len(trade.history)-1].event.price
            datein = bt.num2date(trade.history[0].status.dt)
            dateout = bt.num2date(trade.history[len(trade.history)-1].status.dt)
            if trade.data._timeframe >= bt.TimeFrame.Days:
                datein = datein.date()
                dateout = dateout.date()

            pcntchange = 100 * priceout / pricein - 100
            pnl = trade.history[len(trade.history)-1].status.pnlcomm
            pnlpcnt = 100 * pnl / brokervalue
            barlen = trade.history[len(trade.history)-1].status.barlen
            pbar = pnl / barlen
            self.cumprofit += pnl

            size = value = 0.0
            for record in trade.history:
                if abs(size) < abs(record.status.size):
                    size = record.status.size
                    value = record.status.value

            highest_in_trade = max(trade.data.high.get(ago=0, size=barlen+1))
            lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen+1))
            hp = highest_in_trade - pricein
            lp = lowest_in_trade - pricein
            if dir == 'long':
                mfe0 = hp
                mae0 = lp
                mfe = 100 * hp / pricein
                mae = 100 * lp / pricein
            if dir == 'short':
                mfe0 = -lp
                mae0 = -hp
                mfe = -100 * lp / pricein
                mae = -100 * hp / pricein

            self.trades.append({'ref': trade.ref, 'ticker': trade.data._name,
                'dir': dir, 'datein': datein, 'pricein': pricein,
                'dateout': dateout, 'priceout': priceout,
                 'chng%': round(pcntchange, 2), 'pnl': pnl,
                 'pnl%': round(pnlpcnt, 2), 'size': size, 'value': value,
                 'cumpnl': self.cumprofit, 'nbars': barlen,
                 'pnl/bar': round(pbar, 2), 'mfe': round(mfe0, 2),
                 'mae': round(mae0, 2), 'mfe%': round(mfe, 2),
                 'mae%': round(mae, 2)})        








@njit
def cleaning_loop(entrysum, exitsum):
    """
    
    Loop over buy sell orders given a lag.
    Number of buy and sell should be the same, but shifted.
    The function uses numba: it is compiled and runs in machine code
    
    """
    currenttrade = np.zeros(len(entrysum))
    currenttrade[0] = entrysum[0]
    n = entrysum.shape[0]
    for i in range(1,n):
        # check current trade before, if zero fetch whatever entrysignal is 
        if currenttrade[i-1] == 0:
            currenttrade[i] = entrysum[i]
        # otherwise check if current trade has ended now, --> no current trade
        elif currenttrade[i-1] == exitsum[i]:
            currenttrade[i] = 0
        # else, current trade stays
        else:
            currenttrade[i] = currenttrade[i-1]
    return currenttrade

def cleaninorder(entries, exits):
    """
    
    cleans entries and exits where each exit corresponds to entry in order of appearence.
    Use numba inside the loop
    
    """
    x = pd.DataFrame()
    x['entries'] = entries
    x['exits'] = exits
    x['entrysignal'] = 1 * x['entries'] 
    x['exitsignal']  = 1 * x['exits']
    x['entrysignalsum'] = 0
    x['entrysignalsum'][x['entrysignal']==1] = x['entrysignal'].cumsum()
    x['exitsignalsum'] = 0
    x['exitsignalsum'][x['exitsignal']==1] = x['exitsignal'].cumsum()
    x['currenttrade'] = cleaning_loop(x['entrysignalsum'].values, x['exitsignalsum'].values)
    x['diff'] = x['currenttrade'].diff().fillna(x['currenttrade'][0])
    x['newentries'] = (x['diff']>0)
    x['newexits'] = (x['diff']<0)
    return x['newentries'],x['newexits']

def cleaninorder_old(entries, exits):
    """
    
    cleans entries and exits where each exit corresponds to entry in order of appearence
    
    """
    x = pd.DataFrame()
    x['entries'] = entries
    x['exits'] = exits
    x['entrysignal'] = 1 * x['entries'] 
    x['exitsignal']  = 1 * x['exits']
    x['entrysignalsum'] = 0
    x['entrysignalsum'][x['entrysignal']==1] = x['entrysignal'].cumsum()
    x['exitsignalsum'] = 0
    x['exitsignalsum'][x['exitsignal']==1] = x['exitsignal'].cumsum()
    x['currenttrade'] = 0
    for i in range(len(x)):
        # check current trade before, if zero fetch whatever entrysignal is 
        if (i == 0) or (x['currenttrade'][np.max([i-1,0])] == 0):
            x['currenttrade'][i] = x['entrysignalsum'][i]
        # otherwise check if current trade has ended now, --> no current trade
        elif x['currenttrade'][np.max([i-1,0])] == x['exitsignalsum'][i]:
            x['currenttrade'][i] = 0
        # else, current trade stays
        else:
            x['currenttrade'][i] = x['currenttrade'][i-1]
    x['diff'] = x['currenttrade'].diff().fillna(x['currenttrade'][0])
    x['newentries'] = (x['diff']>0)
    x['newexits'] = (x['diff']<0)
    return x['newentries'],x['newexits']


#entry after signal before N hours, after momentum rsi(Nminutes,rsi_val), monetum EWM
#exit ma,rsi, Nminutes , sl/tp heatmap , ts 

def runsignallight(price,signal,n=100,short=True,fees=0.0007):
    signal=signal.dropna().astype(int)
    if short:
        dp=price['c']/price['c'].shift(-n)-1-fees
    else:
        dp=price['c'].shift(-n)/price['c']-1-fees
    dp=np.sign(dp).replace(-1,0)
    p=dp.loc[signal[signal==1].index].mean()
    return p

def runsignal(price,signal,tp=6,sl=2,ts=5,n1=1,n2=100,rsil=30,rsih=50,rsilag=14,maf=5,mas=15,short=False,size=np.inf,fees=0.07/100,freq='1Min',init_cash=10000,request=None):
    
    res = {}

    if request is None:
        request = []

    runma, runrsi, runn, runts, runtpts = True, True, True, True, True
    if maf is None or mas is None:
        runma = False
    if rsil is None or rsih is None:
        runrsi = False
    if n1 is None or n2 is None:
        runn = False
    if ts is None:
        runts = False
        runtpts = False
        
    #clean price
    price=price.resample(freq).last()

    signalclean=pd.Series.vbt.signals.empty_like(price['c'])
    signalclean.loc[signal[signal].index]=True            
    signal=signalclean
    signal=signal.ffill()
    
    SHORT=short

    rsibelow=rsil
    rsiabove=rsih
    rsiperiod=rsilag

    pf_kwargs = dict(size=size, fees=fees, freq=freq,init_cash=init_cash) #bidask spred 1bps    3bps cdc

    fastmawindow=maf
    slowmawindow=mas

    if runn:
        entryn = n1  # entry after n periods
        exitn = n2  # exit after n periods after entry
        entriesn=signal.shift(entryn).fillna(False)
        exitsnn=entriesn.shift(exitn).fillna(False)
        entriesnn_clean,exitsnn_clean=cleaninorder(entriesn,exitsnn)

    if runma:
        fastma=vbt.MA.run(price['c'], window=fastmawindow,ewm=True)
        slowma=vbt.MA.run(price['c'], window=slowmawindow,ewm=True)

        if not SHORT:
            entriesma = fastma.ma_above(slowma, crossover=True)&signal
            exitsma = fastma.ma_below(slowma, crossover=True)
        elif SHORT:
            entriesma = fastma.ma_below(slowma, crossover=True)&signal
            exitsma = fastma.ma_above(slowma, crossover=True)

        if runn:
            exitsman=entriesma.shift(exitn).fillna(False)
            entriesman_clean,exitsman_clean=cleaninorder(entriesma,exitsman)
    
    if runrsi:
        rsi=vbt.RSI.run(price['c'],window=rsiperiod)
        if not SHORT:
            entriesrsi = rsi.rsi_above(rsiabove, crossover=True)&signal
            exitsrsi = rsi.rsi_below(rsibelow, crossover=True)

        else:
            entriesrsi = rsi.rsi_below(rsibelow, crossover=True)&signal
            exitsrsi = rsi.rsi_above(rsiabove, crossover=True)

        if runn:
            exitsrsi_n=entriesrsi.shift(exitn).fillna(False)
            entriesrsin_clean,exitsrsin_clean=cleaninorder(entriesrsi,exitsrsi_n)

    pf={}
    if not SHORT:
        if runma:
            pf['ma_ma'] = vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesma, exits=exitsma, **pf_kwargs)
            if runts:
                pf['ma_ts'] = vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesma, exits=False, sl_stop=ts/100.,sl_trail=True , **pf_kwargs)
            if runtpts:
                pf['ma_tpts'] = vbt.Portfolio.from_signals(close=price['c'], open=price['o'], high=price['h'], low=price['l'], entries=entriesma, exits=False, sl_stop=ts/100., sl_trail=True, tp_stop= tp/100., **pf_kwargs)
            pf['ma_tpsl']  = vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesma, exits=None, sl_stop=sl/100.,tp_stop=tp/100.,**pf_kwargs)
            if runn:
                pf['ma_n']  = vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesman_clean, exits=exitsman_clean, **pf_kwargs)
        if runrsi:
            pf['rsi_rsi'] =vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesrsi, exits=exitsrsi, **pf_kwargs)
            if runts:
                pf['rsi_ts']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesrsi, exits=False,sl_stop=ts/100.,sl_trail=True ,  **pf_kwargs)
            if runtpts:
                pf['rsi_tpts'] = vbt.Portfolio.from_signals(close=price['c'], open=price['o'], high=price['h'],low=price['l'], entries=entriesrsi, exits=False,sl_stop=ts/100., sl_trail=True, tp_stop= tp/100., **pf_kwargs)
            pf['rsi_tpsl']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesrsi, exits=False,sl_stop=sl/100.,tp_stop=tp/100.,**pf_kwargs)
            if runn:
                pf['rsi_n']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesrsin_clean, exits=exitsrsin_clean,**pf_kwargs)
        if runn:
            pf['n_n']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesnn_clean, exits=exitsnn_clean, **pf_kwargs)
            if runts:
                pf['n_ts']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesn, exits=False,sl_stop=ts/100.,sl_trail=True , **pf_kwargs)
            if runtpts:
                pf['n_tpts']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesn, exits=False,sl_stop=ts/100.,tp_stop=tp/100. ,sl_trail=True , **pf_kwargs)
            pf['n_tpsl']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], entries=entriesn, exits=False,sl_stop=sl/100.,tp_stop=tp/100. , **pf_kwargs)
    else:
        if runma:
            pf['ma_ma'] = vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesma, short_exits=exitsma, **pf_kwargs)
            if runts:
                pf['ma_ts'] = vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesma, short_exits=False, sl_stop=ts/100.,sl_trail=True , **pf_kwargs)
            if runtpts:
                pf['ma_tpts'] = vbt.Portfolio.from_signals(close=price['c'], open=price['o'], high=price['h'], low=price['l'],short_entries=entriesma, short_exits=False, sl_stop=ts/100.,sl_trail=True, tp_stop = tp/100., **pf_kwargs)
            pf['ma_tpsl']  = vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesma, short_exits=None, sl_stop=sl/100.,tp_stop=tp/100.,**pf_kwargs)
            if runn:
                pf['ma_n']  = vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesman_clean, short_exits=exitsman_clean,**pf_kwargs)
        if runrsi:
            pf['rsi_rsi'] =vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesrsi, short_exits=exitsrsi, **pf_kwargs)
            if runts:
                pf['rsi_ts']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesrsi, short_exits=False,sl_stop=ts/100.,sl_trail=True ,  **pf_kwargs)
            if runtpts:
                pf['rsi_tpts'] = vbt.Portfolio.from_signals(close=price['c'], open=price['o'], high=price['h'],low=price['l'], short_entries=entriesrsi, short_exits=False,sl_stop=ts/100., tp_stop = tp/100., sl_trail=True, **pf_kwargs)
            pf['rsi_tpsl']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesrsi, short_exits=False, sl_stop=sl/100.,tp_stop=tp/100.,**pf_kwargs)
            if runn:
                pf['rsi_n']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesrsin_clean, short_exits=exitsrsin_clean, **pf_kwargs)
        if runn:
            pf['n_n']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesnn_clean, short_exits=exitsnn_clean, **pf_kwargs)
            if runts:
                pf['n_ts']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesn, short_exits=False,sl_stop=ts/100.,sl_trail=True ,**pf_kwargs)
            if runtpts:
                pf['n_tpts']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesn, short_exits=False,sl_stop=ts/100.,tp_stop = tp/100., sl_trail=True ,**pf_kwargs)
            pf['n_tpsl']=vbt.Portfolio.from_signals(close=price['c'],open=price['o'],high=price['h'],low=price['l'], short_entries=entriesn, short_exits=False, sl_stop=sl/100.,tp_stop=tp/100.,**pf_kwargs)
    

    if 'pf' in request:
        for k in pf:
            res[k+'_pf']=pf[k]

    if 'backtest' in request:
        return res

    dfs=[]
    for k in pf:
        dfstats=DF(pf[k].stats())
        dfstats['k']=k
        dfstats=dfstats.set_index('k')
        dfs.append(dfstats)

    for k in pf:
        dftr=pf[k].orders.records_readable
        dftr['k']=k
        dfs.append(dftr)
    
    for k in pf:
        dftr=pf[k].trades.records_readable
        dftr['k']=k
        dfs.append(dftr)

    df = DF([{'k': k, 'sr': pf[k].sharpe_ratio(), 'tr': pf[k].total_return()} for k in pf.keys()])

    res['df']=df
    res['dfs']=dfs
        
    return res


def wrapped_runsignal(params_and_data_dict):
    param_dict = params_and_data_dict.copy()
    args = inspect.signature(runsignal).parameters.keys()
    param_dict['ts'] = param_dict['sl']
    param_dict_passed = dict((k, param_dict[k]) for k in args)
    ressig=runsignal(**param_dict_passed)

    res=[]
    del param_dict['price']
    del param_dict['signal']
    del param_dict['fees']
    del param_dict['freq']
    del param_dict['init_cash']
    del param_dict['request']
    del param_dict['signalid']
    del param_dict['side']

    for k in ressig: # put all keys blow 
        ### remove price and signal
        resd = param_dict.copy()
        resd['k']=k
        resd['sr']=ressig[k].sharpe_ratio()
        resd['tr']=ressig[k].total_return()
        resd['ntrades'] = ressig[k].trades.count()/param_dict['num_weeks']
        res.append(resd)
    return res
