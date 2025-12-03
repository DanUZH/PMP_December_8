# Functions for Backtesting

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import statsmodels.api as sm


def summarizePerformance(xsReturns, Rf, factorXsReturns, 
    annualizationFactor, strategyNames, fileName):
    
    nPeriods = xsReturns.shape[0]
    nAssets = 1 if (xsReturns.ndim == 1) else xsReturns.shape[1]
    nFactors = 1 if (factorXsReturns.ndim == 1) else factorXsReturns.shape[1]
    totalReturns = xsReturns + Rf
    
    # Compute the terminal value of the portfolios to get the geometric mean
    # return per period
    FinalPfValRf = np.prod(1 + Rf)
    FinalPfValTotalRet = np.prod(1 + totalReturns, axis=0)
    GeomAvgRf = 100 * (FinalPfValRf**(annualizationFactor / nPeriods) - 1)
    GeomAvgTotalReturn = 100 * (FinalPfValTotalRet**(annualizationFactor / nPeriods) - 1)
    GeomAvgXsReturn = GeomAvgTotalReturn - GeomAvgRf

    # Regress returns on benchmark to get alpha, factor exposures, and tracking error
    alphaArithmetic = annualizationFactor * 100 * np.ones((1, nAssets))
    alphaPVal = np.zeros((1, nAssets))
    betas = np.zeros((nFactors, nAssets))
    trackingError = 100 * np.sqrt(annualizationFactor) * np.ones((1, nAssets))
    x = factorXsReturns
    # Add a constant to the explanatory variables
    x = sm.add_constant(x)
    for asset in range(nAssets):
        y = xsReturns[:, asset]
        # Estimate OLS with robust standard errors
        model = sm.OLS(y, x).fit(cov_type = 'HAC', cov_kwds={'maxlags':1})
        alphaArithmetic[0, asset] *= model.params[0]
        alphaPVal[0, asset] = model.pvalues[0]
        betas[:, asset] = model.params[1:]
        trackingError[0, asset] *= np.sqrt(model.mse_resid)
        if (abs(alphaArithmetic[0, asset]) < 1e-12):
            alphaArithmetic[0, asset] = 0
            alphaPVal[0, asset] = 1

    # Based on the regression estimates, compute the total return on the passive 
    # alternative and the annualized alpha
    bmRet = factorXsReturns @ betas + Rf
    FinalPfValBm = np.prod(1 + bmRet, axis=0)
    GeomAvgBmReturn = 100 * (FinalPfValBm**(annualizationFactor / nPeriods) - 1)
    alphaGeometric = GeomAvgTotalReturn - GeomAvgBmReturn
    for asset in range(nAssets):
        if (abs(alphaGeometric[asset]) < 1e-12):
            alphaGeometric[asset] = 0
    
    InfoRatio = alphaGeometric / trackingError

    # Rescale the returns to be in percentage points
    xsReturns *= 100
    totalReturns *= 100
    ArithmAvgTotalReturn = annualizationFactor * np.mean(totalReturns, axis = 0)
    ArithmAvgXsReturn = annualizationFactor * np.mean(xsReturns, axis = 0)
    StdXsReturns = np.sqrt(annualizationFactor) * np.std(xsReturns, axis = 0, ddof = 1)
    SharpeArithmetic = ArithmAvgXsReturn / StdXsReturns
    SharpeGeometric = GeomAvgXsReturn / StdXsReturns

    MinXsReturn = np.min(xsReturns, axis = 0)
    MaxXsReturn = np.max(xsReturns, axis = 0)
    SkewXsReturn = skew(xsReturns, axis = 0)
    KurtXsReturn = kurtosis(xsReturns, axis = 0)

    # Compute first three autocorrelations
    AC = np.zeros((3, nAssets))
    for asset in range(nAssets):
        # Could al
        #AC[0, asset] = np.corrcoef(xsReturns[:-1, asset], xsReturns[1:, asset])[0, 1]
        #AC[1, asset] = np.corrcoef(xsReturns[:-2, asset], xsReturns[2:, asset])[0, 1]
        #AC[2, asset] = np.corrcoef(xsReturns[:-3, asset], xsReturns[3:, asset])[0, 1]
        AC[0, asset] = pd.Series(xsReturns[:, asset]).autocorr(lag=1)
        AC[1, asset] = pd.Series(xsReturns[:, asset]).autocorr(lag=2)
        AC[2, asset] = pd.Series(xsReturns[:, asset]).autocorr(lag=3)

    allStats = np.vstack([
        ArithmAvgTotalReturn, ArithmAvgXsReturn, StdXsReturns, SharpeArithmetic, GeomAvgTotalReturn,
        GeomAvgXsReturn, SharpeGeometric, MinXsReturn, MaxXsReturn, SkewXsReturn, KurtXsReturn,
        alphaArithmetic, alphaGeometric, betas, trackingError, InfoRatio, AC
    ])

    betaNames = [f'Beta {i}' for i in range(1, betas.shape[0] + 1)]

    rowNames = [
        'Arithm Avg Total Return', 'Arithm Avg Xs Return', 'Std Xs Returns', 'Sharpe Arithmetic',
        'Geom Avg Total Return', 'Geom Avg Xs Return', 'Sharpe Geometric', 'Min Xs Return', 'Max Xs Return',
        'Skewness', 'Excess Kurtosis', 'Alpha Arithmetic', 'Alpha Geometric'
        ] + betaNames + ['Tracking Error', 'Information Ratio', 'AC 1', 'AC 2', 'AC 3']

    t = pd.DataFrame(allStats, index=rowNames, columns=strategyNames)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(t)

    # The mess below is to left-align the first column.
    # Without left-alignment just say t.to_excel(fileName, index=True, float_format='%.4f')
    from openpyxl.styles.alignment import Alignment
    with pd.ExcelWriter(fileName, engine='openpyxl') as writer:
        sht_name = 'Performance Statistics'
        t.to_excel(writer, index=True, sheet_name=sht_name, float_format='%.4f')
        sheet = writer.sheets[sht_name]
        for cell in sheet['A']:
            cell.alignment = Alignment(horizontal='left')
