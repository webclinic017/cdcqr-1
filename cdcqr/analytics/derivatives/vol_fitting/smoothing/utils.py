import sys
sys.path.insert(1,"/core/github/cryptoderiv-quant-lib")
import pandas as pd
import numpy as np
from cryptoderiv_quantlib import VolModels
from cryptoderiv_quantlib import VolCurve, VolSurface, StickyOptions



def cumsum_getTEvents(gRaw,h):
    tEvents = []
    sPos = 0
    sNeg = 0
    diff=gRaw.diff()
    for i in diff.index[1:]:
        sPos = max(0,sPos+diff.loc[i])
        sNeg = min(0,sNeg+diff.loc[i])
        if sNeg<-h:
            sNeg=0;
            tEvents.append(i)
        elif sPos>h:
            sPos=0;
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


