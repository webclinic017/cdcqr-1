import sys
sys.path.insert(1,"/core/github/cryptoderiv-quant-lib")
import pandas as pd
import numpy as np
from cryptoderiv_quantlib import VolModels
from scipy.interpolate import interp1d


def cumsum_getTEvents(gRaw,h):
    tEvents = []
    sPos = 0
    sNeg = 0
    diff=gRaw.diff()
    for i in diff.index[1:]:
        sPos = max(0,sPos+diff.loc[i])
        sNeg = min(0,sNeg+diff.loc[i])
        if sNeg<-h:
            sNeg=0
            tEvents.append(i)
        elif sPos>h:
            sPos=0
            tEvents.append(i)
    # return pd.DatetimeIndex(tEvents)
    return tEvents


def fit_fun2(curve,df,useprevATMvol=True):
    x = [curve.params['skew_left'], curve.params['skew_right'], 
         curve.params['conv_left'], curve.params['conv_right'], 
         curve.params['rightwing'], curve.params['leftwing'], curve.params['vol_atm']]
        
    return fit_fun3(x,df,useprevATMvol=True)
    
def fit_fun3(x,df,useprevATMvol=True, MODEL_TO_TEST = "parabolic_linear6"):
    
    model = VolModels.vol_model(MODEL_TO_TEST)
    skewleft=x[0]
    skewright=x[1]
    leftconv=x[2]
    rightconv=x[3]
    rightwing=x[4]
    leftwing=x[5]
    df['m'] = 0.5*(df['aiv']+df['biv'])
    S=df['s'].iloc[0]
    if useprevATMvol:
        volATM = x[6]
    else:
        volATM=interp1d(df.index, df['m'])([S])[0]
    kwargs = model._default_vol_model_config()
    kright=kwargs['kright']
    kleft=kwargs['kleft']
    
    df.loc[(df.logKFtau>=0)&(df.logKFtau<kright),'myvol']=volATM+(skewright*df['logKFtau']+rightconv*df['logKFtau']**2)
    df.loc[(df.logKFtau<0)&(df.logKFtau>kleft),'myvol']=volATM+(-skewleft*df['logKFtau']+leftconv*df['logKFtau']**2)
    yright=volATM+(skewright*kright+rightconv*kright**2)
    yleft=volATM+(-skewleft*kleft+leftconv*kleft**2)
    df.loc[(df.logKFtau>=kright),'myvol']=rightwing*(df['logKFtau']-kright)+yright
    df.loc[(df.logKFtau<=kleft),'myvol']=-leftwing*(df['logKFtau']-kleft)+yleft
    df['insidebidask']=False
    df.loc[(df.myvol>df.biv)&(df.myvol<df.aiv),'insidebidask']=True
    
    return df['insidebidask']


def err_fun1(curve,dftTOTM):
    x = [curve.params['skew_left'], curve.params['skew_right'], 
         curve.params['conv_left'], curve.params['conv_right'], 
         curve.params['rightwing'], curve.params['leftwing']]
    MODEL_TO_TEST = "parabolic_linear6"
    model = VolModels.vol_model(MODEL_TO_TEST)
    return np.sqrt(model._err_fun(x, usevega=True, df=dftTOTM, **model._default_vol_model_config()))
    

def err_fun2(curve,df,useprevATMvol=True):
    x = [curve.params['skew_left'], curve.params['skew_right'], 
         curve.params['conv_left'], curve.params['conv_right'], 
         curve.params['rightwing'], curve.params['leftwing'], curve.params['vol_atm']]
        
    return err_fun3(x,df,useprevATMvol=True)
    
def err_fun3(x,df,useprevATMvol=True):
    skewleft=x[0]
    skewright=x[1]
    leftconv=x[2]
    rightconv=x[3]
    rightwing=x[4]
    leftwing=x[5]
    
    df['m'] = 0.5*(df['aiv']+df['biv'])
    S=df['s'].iloc[0]
    if useprevATMvol:
        volATM = x[6]
    else:
        volATM=interp1d(df.index, df['m'])([S])[0]

    MODEL_TO_TEST = "parabolic_linear6"
    model = VolModels.vol_model(MODEL_TO_TEST)
    kwargs = model._default_vol_model_config()
    kright=kwargs['kright']
    kleft=kwargs['kleft']
    
    df.loc[(df.logKFtau>=0)&(df.logKFtau<kright),'myvol']=volATM+(skewright*df['logKFtau']+rightconv*df['logKFtau']**2)
    df.loc[(df.logKFtau<0)&(df.logKFtau>kleft),'myvol']=volATM+(-skewleft*df['logKFtau']+leftconv*df['logKFtau']**2)
    yright=volATM+(skewright*kright+rightconv*kright**2)
    yleft=volATM+(-skewleft*kleft+leftconv*kleft**2)
    df.loc[(df.logKFtau>=kright),'myvol']=rightwing*(df['logKFtau']-kright)+yright
    df.loc[(df.logKFtau<=kleft),'myvol']=-leftwing*(df['logKFtau']-kleft)+yleft
    df['insidebidask']=False
    df.loc[(df.myvol>df.biv)&(df.myvol<df.aiv),'insidebidask']=True
    
    res0=(df['myvol']-df['m'])**2
    
    usevega = True
    if usevega:
        if df['vega'].sum() != 0:
            res0*=df['vega']/df['vega'].sum()

    return np.sqrt(res0.mean())


def optchain2local_vol_res_df(df0):
    df1 = df0[['tm','expire','strike','ty','s','aiv','biv','vega']].reset_index(drop=True).sort_values(by=['tm','expire','strike']).drop_duplicates().reset_index(drop=True)
    df1['params'] = [{} for i in range(len(df1))]
    df1['tm'] = [int(df1['tm'][i].timestamp()*1000) for i in range(len(df1))]
    df1['expire'] = [int(df1['expire'][i].timestamp()*1000) for i in range(len(df1))]
    df = df1.set_index(['tm','expire','strike'])
    
    MODEL_TO_TEST = "parabolic_linear6"
    model = VolModels.vol_model(MODEL_TO_TEST)
    
    ts = df.index.get_level_values("tm").drop_duplicates()
    res2 = []
    last_valid_curve = None
    thres1 = 0.002
    thres2 = 0.0015
    cooldown_flag = False
    change_curve_counter = 0 

    for t in ts:
        dft = df.loc[[t]].droplevel("tm")
        Ts = dft.index.get_level_values("expire").drop_duplicates()
        for T in Ts:
            #print(t, T, len(res), len(res2))
            dftT = None
            try:
                dftT = dft.loc[[T]].droplevel("expire")
                tau = (T - t) / (1000 * 60 * 60 * 24 * 365)
                dftT["logKFtau"] = np.log(dftT.index / dftT["s"]) / np.sqrt(tau)
                S = dftT["s"].iloc[0]
                dftTOTM2 = dftT[
                              ((dftT.ty == 1) & (S < dftT.index))
                              | ((dftT.ty == -1) & (S > dftT.index))
                              ]

                if len(dftTOTM2) == 0:
                    continue

                __fitted_vol_pl6 = model._trigger_vol_fit(df.loc[([t],[T]),:], plot=False, vol_model_config_in={
                                                          "eps": 1e-09,
                                                          "maxfun": 100,
                                                          "maxiter": 100,
                                                          "kright": 1.5,
                                                          "kleft": -1.5,
                                                        })
                #print(dftTOTM2)
                err1 = err_fun1(__fitted_vol_pl6.curves[0], dftTOTM2)
                curr_err = err1
                #print(err1)
                #res.append([t, T, dftTOTM2, 
                #            __fitted_vol_pl6.curves[0], err1]) 
                #print("x")

                if last_valid_curve is None:
                    last_valid_curve = __fitted_vol_pl6.curves[0]
                err2 = err_fun2(last_valid_curve, dftTOTM2)

                if cooldown_flag:
                    if err1 <= thres2:
                        # use new curve
                        last_valid_curve = __fitted_vol_pl6.curves[0]
                        curr_err = err1 
                        cooldown_flag = False
                        change_curve_counter = change_curve_counter +1 

                    else:
                        # keep the last_valid_curve
                        curr_err = err2

                else: # not on cooldown, use high threshold
                    if err2 <= thres1:
                        # keep the last_valid_curve
                        curr_err = err2 

                    elif err1 > thres1:
                        # freeze the last valid curve, wait for cooldown 
                        curr_err = err2
                        cooldown_flag = True

                    else:
                        # use new curve
                        last_valid_curve = __fitted_vol_pl6.curves[0]
                        curr_err = err1
                        change_curve_counter = change_curve_counter +1 

                res2.append([t, T, dftTOTM2, __fitted_vol_pl6.curves[0],
                             last_valid_curve, 
                             err1, err2, curr_err,      # 5,6,7
                             change_curve_counter 
                            ])
                #print('y')

            except Exception as e: 
                print(e)
                
    return pd.DataFrame(res2, columns=['t','expire','df','localcurve','currcurve','err1','err2','currerr','changecurvecounter'])
