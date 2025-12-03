
##### PMP FUNDAMENTAL FUNCTIONS V4 #####

#-----------------------------------------------------------------------------------------------------------------------------------------------
# SIGNAL / WEIGHTS CREATION --------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

def make_country_weights(
    signal,
    returns,
    benchmark_series,
    k=1,
    long_short=False,
    beta_neutral=False,
    signal_lag=0,
    min_regions=1,
    beta_window=12,
):
    """
    CASE A: long_short=False 
        → Long-only Top-k, sum(w)=1
        
    CASE B: long_short=True & beta_neutral=False 
        → Dollar-neutral (sum(w)=0), self-financing
        
    CASE C: long_short=True & beta_neutral=True
        → Long/Short, beta-neutral (gross exposure = 1)
          Using shrinkage beta:
              Beta = 0.2 + 0.8 * (cov / var)
    """

    import pandas as pd
    import numpy as np

    # ---------- 1) Signal ranks (with lag) ----------
    if signal_lag > 0:
        signal = signal.shift(signal_lag)
    else:
        signal = signal

    ranks = signal.rank(axis=1, ascending=False)

    idx = ranks.index.union(returns.index)
    ranks = ranks.reindex(idx)
    returns = returns.reindex(idx)

    regions = ranks.columns
    ret_next = returns.shift(-1)
    eligible = ranks.notna() & ret_next.notna()
    valid_dates = eligible.sum(axis=1) >= min_regions

    # ---------- 2) Compute shrinkage betas ----------
    if long_short and beta_neutral:
        bm = benchmark_series.reindex(returns.index)

        betas = {}
        for r in regions:
            cov = returns[r].rolling(beta_window).cov(bm)
            var = bm.rolling(beta_window).var()
            raw_beta = cov / var

            # Shrinkage beta: 0.2 + 0.8 * raw_beta
            shrink_beta = 0.2 + 0.8 * raw_beta

            betas[r] = shrink_beta

        betas = pd.DataFrame(betas)

    # ---------- 3) Build weights ----------
    out = []

    for t in ranks.index[valid_dates]:

        usable = eligible.loc[t]
        cols = usable[usable].index
        r_t = ranks.loc[t, cols].dropna()
        n = len(r_t)

        if n < min_regions:
            continue

        w = pd.Series(0.0, index=regions)

        # =======================
        # CASE A — Long-only
        # =======================
        if not long_short:
            winners = r_t.nsmallest(min(k, n)).index
            w[winners] = 1 / len(winners)
            out.append(w.rename(t))
            continue

        # =======================
        # CASE B/C — Long-short
        # =======================
        k_eff = min(k, n // 2)
        winners = r_t.nsmallest(k_eff).index
        losers  = r_t.nlargest(k_eff).index

        w[winners] = +1
        w[losers]  = -1

        # ---------- CASE B: Dollar-neutral ----------
        if not beta_neutral:
            # sum(w)=0
            w = w - w.mean()

            # normalize gross exposure = 1
            gross = w.abs().sum()
            if gross > 0:
                ### CHANGE HERE FOR GROSS EXPOSURE = 2
                #w /= gross
                w = w / (gross/2)

            out.append(w.rename(t))
            continue

        # =======================
        # CASE C — Beta-neutral
        # =======================
        beta_t = betas.loc[t]

        # Portfolio beta (using shrinkage beta)
        beta_p = (w * beta_t).sum()

        # Hedge by adjusting the benchmark region weight
        hedge_w = -beta_p
        w[benchmark_series.name] += hedge_w

        # Scale to gross exposure = 1
        gross = w.abs().sum()
        if gross > 0:
            w /= gross

        out.append(w.rename(t))

    return pd.DataFrame(out).fillna(0)


def append_final_zero_row(weights, final_date="2025-10-31"):
    """
    Append a final row of zeros to the weights DataFrame.
    
    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights indexed by date.
    
    final_date : str or Timestamp
        Date at which a zero-weight row should be added.
    
    Returns
    -------
    pd.DataFrame
        Original weights with an additional final zero row.
    """
    import pandas as pd

    # Convert final_date into a Timestamp (in case user provides a string)
    final_date = pd.Timestamp(final_date)

    # Create a row of zeros matching all columns (regions/assets)
    zero_row = pd.Series(0.0, index=weights.columns, name=final_date)

    # Append it to the existing DataFrame
    weights_extended = pd.concat([weights, zero_row.to_frame().T])

    return weights_extended


#-----------------------------------------------------------------------------------------------------------------------------------------------
# BACKTEST FUNCTIONS ---------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


def run_timing_strategy(
    signal,
    returns,
    rf,
    frequency=12,
    t_cost=0.0,
    fin_rate_annual=0,
):
    """
    Runs a timing-based long/short strategy using:
        - A predictive signal (determines market weight)
        - Market total returns
        - Risk-free returns
        - Optional transaction costs (t_cost per 1.0 turnover)
        - Optional financing cost on borrowed cash (w_rf < 0)

    Financing:
        • If w_rf < 0 (we borrow cash), we pay fin_rate_annual p.a.
        • If w_rf >= 0, no financing cost is applied.
    """

    import pandas as pd
    import numpy as np

    # assume monthly data → 12 periods per year
    periods_per_year = 12
    # you can also use: (1 + fin_rate_annual)**(1/12) - 1
    fin_rate_per_period = (1+fin_rate_annual)**(1/periods_per_year) -1


    # ----------------------------------------------------------------------
    # 1. Clean and align the signal and returns
    # ----------------------------------------------------------------------
    signal = signal.copy().fillna(0)
    signal.name = "signal"

    dates = list(signal.index)

    mkt_ret = returns.reindex(dates).fillna(0)
    rf_ret = rf.reindex(dates).ffill().fillna(0)

    # ----------------------------------------------------------------------
    # 2. Rebalancing schedule
    # ----------------------------------------------------------------------
    rebalance = pd.Series(0, index=dates, dtype=int)
    rebalance.iloc[::frequency] = 1      # periodic rebalancing
    rebalance.iloc[0] = 1                # always rebalance at t=0

    # ----------------------------------------------------------------------
    # 3. Initial portfolio: 100% risk-free
    # ----------------------------------------------------------------------
    current_weight = pd.Series([0.0, 1.0], index=["mkt", "rf"])

    results = []

    # ----------------------------------------------------------------------
    # 4. Main backtest loop
    # ----------------------------------------------------------------------
    for i in range(len(dates) - 1):

        date = dates[i]
        next_date = dates[i + 1]

        # ---------------------------------------------------------
        # STEP 1 — Rebalancing at the end of month t
        # ---------------------------------------------------------
        if rebalance.loc[date] == 1:
            sig = float(signal.loc[date])
            target_weight = pd.Series([sig, 1.0 - sig], index=["mkt", "rf"])

            turnover = 0.5 * (target_weight - current_weight).abs().sum()
            current_weight = target_weight.copy()
        else:
            turnover = 0.0

        # ---------------------------------------------------------
        # STEP 2 — Apply returns for next month (t+1)
        # ---------------------------------------------------------
        r_vec = pd.Series(
            [mkt_ret.loc[next_date], rf_ret.loc[next_date]],
            index=["mkt", "rf"],
        )

        # Portfolio gross return before any costs
        gross_ret = float((current_weight * r_vec).sum())

        # --- Trading cost on turnover ---
        tcost = turnover * t_cost

        # --- Financing cost only if rf weight is negative (borrowing) ---
        borrowed_share = max(-current_weight["rf"], 0.0)  # e.g. rf = -0.3 → borrow 0.3
        fcost = borrowed_share * fin_rate_per_period

        # Net return after all costs
        net_ret = gross_ret - tcost - fcost

        # Benchmark = market-only return
        bm_ret = float(mkt_ret.loc[next_date])

        # ---------------------------------------------------------
        # STEP 3 — Drift update
        # ---------------------------------------------------------
        new_value = current_weight * (1.0 + r_vec)
        current_weight = new_value / new_value.sum()

        # ---------------------------------------------------------
        # STEP 4 — Save results
        # ---------------------------------------------------------
        results.append(
            {
                "Date": next_date,
                "ret_net": net_ret,
                "ret_gross": gross_ret,
                "turnover": turnover,
                "tcost": tcost,
                "fcost": fcost,
                "w_mkt": current_weight["mkt"],
                "w_rf": current_weight["rf"],
                "ret_bm": bm_ret,
                "ret_rf": float(rf_ret.loc[next_date]),
            }
        )

    return pd.DataFrame(results).set_index("Date")


#-----------------------------------------------------------------------------------------------------------------------------------------------


def run_cc_strategy(weights, returns, rf, frequency=1, t_cost=0.0,
                    benchmark="equal", long_short=False, beta_neutral=False):
    """
    Cross-country backtest engine.

    Supports:
      • Long-only allocation
      • Long/short allocation
      • Benchmark as:
            - string: "equal", "none", or region
            - pd.Series: benchmark returns
            - pd.DataFrame: benchmark weights per region

    Parameters
    ----------
    weights : pd.DataFrame
        Zielgewichte pro Rebalancing (Index = Dates, Columns = Regionen).

    returns : pd.DataFrame
        Monatliche Total Returns für dieselben Regionen.

    rf : pd.Series
        Monatlicher Risk-Free Return.

    frequency : int
        Rebalancing-Intervall in Monaten (1 = monatlich).

    t_cost : float
        Transaktionskosten pro 1.0 Einheit Turnover.

    benchmark : str, Series, DataFrame
        Benchmark Definition (neu: Series oder DataFrame möglich).

    long_short : bool
        True  → long/short normalization
        False → long-only normalization

    beta_neutral : bool
        • Only relevant if long_short=True.
        • Determines drift normalisation:
            - long_short=True, beta_neutral=False → dollar-neutral drift
            - long_short=True, beta_neutral=True  → gross-exposure drift (Sum|w| = 1)

    Returns
    -------
    results : pd.DataFrame
    """

    import pandas as pd
    import numpy as np

    # ------------------------------------------------------------
    # 1. Align inputs
    # ------------------------------------------------------------
    weights = weights.copy().fillna(0)
    returns = returns.reindex(weights.index).copy()
    returns = returns[weights.columns]
    rf = rf.reindex(weights.index).fillna(0)

    dates = list(weights.index)
    regions = list(weights.columns)

    # ------------------------------------------------------------
    # 2. Rebalancing schedule
    # ------------------------------------------------------------
    rebalance = pd.Series(0, index=dates, dtype=int)
    rebalance.iloc[::frequency] = 1
    rebalance.iloc[0] = 1

    # ------------------------------------------------------------
    # 3. Benchmark handling (unchanged)
    # ------------------------------------------------------------
    if isinstance(benchmark, str):

        if benchmark == "equal":
            benchmark_w = pd.DataFrame(
                1 / len(regions),
                index=weights.index,
                columns=regions
            )

        elif benchmark == "none":
            benchmark_w = pd.DataFrame(
                0,
                index=weights.index,
                columns=regions
            )

        elif benchmark in regions:
            benchmark_w = pd.DataFrame(
                0,
                index=weights.index,
                columns=regions
            )
            benchmark_w[benchmark] = 1.0

        else:
            raise ValueError("Invalid benchmark string")

        bm_returns = (benchmark_w * returns).sum(axis=1)

    elif isinstance(benchmark, pd.Series):
        bm_returns = benchmark.reindex(weights.index).fillna(0)
        benchmark_w = pd.DataFrame(np.nan, index=weights.index, columns=regions)

    elif isinstance(benchmark, pd.DataFrame):
        benchmark_w = benchmark.reindex(index=weights.index, columns=regions).fillna(0)
        bm_returns = (benchmark_w * returns).sum(axis=1)

    else:
        raise ValueError("Benchmark must be str, Series, or DataFrame.")

    # ------------------------------------------------------------
    # 4. Start weights
    # ------------------------------------------------------------
    current_weight = weights.iloc[0].copy()
    results = []

    # ------------------------------------------------------------
    # 5. Backtest Loop
    # ------------------------------------------------------------
    for i in range(len(dates) - 1):

        date = dates[i]
        next_date = dates[i + 1]

        # STEP 1 – Rebalancing
        if rebalance.loc[date] == 1:
            target_weight = weights.loc[date].fillna(0)
            turnover = 0.5 * (target_weight - current_weight).abs().sum()
            current_weight = target_weight.copy()
        else:
            turnover = 0.0

        # STEP 2 – Apply returns
        r_vec = returns.loc[next_date].fillna(0.0)

        gross_ret = (current_weight * r_vec).sum()
        tcost = turnover * t_cost
        net_ret = gross_ret - tcost
        bm_ret = bm_returns.loc[next_date]

        # STEP 3 – Drift Update
        new_weight = current_weight * (1 + r_vec)

        # --------------------------------------------------------
        # CASE A: Long-only (sum(w)=1)
        # --------------------------------------------------------
        if not long_short:
            total = new_weight.sum()
            if total != 0:
                current_weight = new_weight / total
            else:
                current_weight = new_weight.copy()

        # --------------------------------------------------------
        # CASE B: Long/Short Dollar-neutral
        # sum(w)=0 & sum(|w|)=1
        # ONLY apply drift to active regions (non-zero)
        # --------------------------------------------------------
        elif long_short and not beta_neutral:

            # Identify active regions in current portfolio
            active = current_weight[current_weight != 0].index

            # Drift only active weights
            new_active = current_weight[active] * (1 + r_vec[active])

            # Enforce dollar-neutral: sum(w)=0
            centered = new_active - new_active.mean()

            # Normalize to sum(|w|)=1
            ### CHANGE HERE FOR GROSS EXPOSURE
            #centered = centered / centered.abs().sum()
            centered = centered / (centered.abs().sum()/2)

            # Rebuild full weight vector
            current_weight = pd.Series(0.0, index=current_weight.index)
            current_weight[active] = centered

        # --------------------------------------------------------
        # CASE C: Long/Short Beta-neutral
        # Sum(|w|)=1
        # --------------------------------------------------------
        else:  # long_short=True & beta_neutral=True
            gross = new_weight.abs().sum()
            if gross != 0:
                current_weight = new_weight / gross
            else:
                current_weight = new_weight.copy()

        # STEP 4 – Record results
        row = {
            "Date": next_date,
            "ret_net": net_ret,
            "ret_gross": gross_ret,
            "ret_bm": bm_ret,
            "turnover": turnover,
            "tcost": tcost,
            "ret_rf": rf.loc[next_date],
        }

        for reg in regions:
            row[f"w_{reg}"] = current_weight.get(reg, np.nan)

        results.append(row)

    return pd.DataFrame(results).set_index("Date")




#-----------------------------------------------------------------------------------------------------------------------------------------------
# PERFORMANCE STATISTICS -----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


def run_perf_summary_benchmark_vs_strategy(
    results,
    alreadyXs=False,
    annualizationFactor=12,
    strategyNames=["Benchmark", "Strategy"]
):
    """
    Builds benchmark & strategy XS returns and runs summarizePerformance().

    EXACT replica of:
        xs_bm = results["ret_bm"] - results["ret_rf"]
        xs_str = results["ret_net"] - results["ret_rf"] (unless alreadyXs=True)
        xsReturns = np.column_stack([xs_bm, xs_str])
        Rf = results["ret_rf"]
        factorXsReturns = (results["ret_bm"] - results["ret_rf"])
        summarizePerformance(...)
    """

    import numpy as np

    # --------------------------------------
    # 1) Benchmark excess return
    # --------------------------------------
    xs_bm = results["ret_bm"] - results["ret_rf"]

    # --------------------------------------
    # 2) Strategy excess return
    # --------------------------------------
    if alreadyXs:
        xs_str = results["ret_net"]
    else:
        xs_str = results["ret_net"] - results["ret_rf"]

    # --------------------------------------
    # 3) Combine into 2D matrix (benchmark, strategy)
    # --------------------------------------
    xsReturns = np.column_stack([xs_bm, xs_str])

    # --------------------------------------
    # 4) Risk-free (needs 2D: T × 1)
    # --------------------------------------
    Rf = results["ret_rf"].to_numpy().reshape(-1, 1)

    # --------------------------------------
    # 5) Factor XS returns = Benchmark XS return (2D!)
    # --------------------------------------
    factorXsReturns = (results["ret_bm"].to_numpy() - 
                       results["ret_rf"].to_numpy()).reshape(-1, 1)

    # --------------------------------------
    # 6) Performance summary
    # --------------------------------------
    return summarizePerformance(
        xsReturns=xsReturns,
        Rf=Rf,
        factorXsReturns=factorXsReturns,
        annualizationFactor=annualizationFactor,
        strategyNames=strategyNames
    )


#-----------------------------------------------------------------------------------------------------------------------------------------------


def run_factor_regression(
    results,
    factor_data,
    alreadyXs=False,
    annualizationFactor=12,
    strategyNames=["Strategy"]
):
    """
    Builds all required inputs for summarizePerformance() and runs it.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain:
            ret_net - strategy total return
            ret_rf  - risk-free return

    factor_data : pd.DataFrame
        Factor excess returns (or total returns depending on context)

    alreadyXs : bool
        True  -> results["ret_net"] already contains excess returns
        False -> compute xs = ret_net - ret_rf

    annualizationFactor : int
        E.g. 12 for monthly data

    strategyNames : list
        Column names for output table

    Returns
    -------
    pd.DataFrame
        Performance summary
    """

    import pandas as pd
    import numpy as np

    # -----------------------------
    # 1) Compute strategy excess returns
    # -----------------------------
    if alreadyXs:
        xsReturnsStrategy = results["ret_net"]
    else:
        xsReturnsStrategy = results["ret_net"] - results["ret_rf"].values

    xsReturnsStrategy = xsReturnsStrategy.to_frame()   # ensure 2D

    # -----------------------------
    # 2) Merge strategy with factor data (nearest matching date)
    # -----------------------------
    combined = pd.merge_asof(
        xsReturnsStrategy.sort_index(),
        factor_data.sort_index(),
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("5D")
    )

    # factor excess returns (2D)
    factorXsReturns = combined.iloc[:, 1:]

    # -----------------------------
    # 3) Convert everything to numpy for summarizePerformance()
    # -----------------------------
    xsReturns = xsReturnsStrategy.to_numpy().copy()

    Rf = results["ret_rf"].to_numpy().reshape(-1, 1)

    factorXsReturns = factorXsReturns.to_numpy().copy()

    # -----------------------------
    # 4) Run performance summary
    # -----------------------------
    return summarizePerformance(
        xsReturns=xsReturns,
        Rf=Rf,
        factorXsReturns=factorXsReturns,
        annualizationFactor=annualizationFactor,
        strategyNames=strategyNames
    )


#-----------------------------------------------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS -----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


def summarizePerformance(xsReturns, Rf, factorXsReturns, 
    annualizationFactor, strategyNames):
    """
    Summarizes performance statistics for one or multiple strategies.
    Based on the helper package (Ziegler).
    Adds:
        - Alpha arithmetic p-values
        - Beta p-values (for each factor)
    Removes:
        - Excel output
    Returns:
        pd.DataFrame
    """

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy.stats import skew, kurtosis

    nPeriods = xsReturns.shape[0]
    nAssets = 1 if (xsReturns.ndim == 1) else xsReturns.shape[1]
    nFactors = 1 if (factorXsReturns.ndim == 1) else factorXsReturns.shape[1]
    totalReturns = xsReturns + Rf

    # ---------- GEOMETRIC RETURNS ----------
    FinalPfValRf = np.prod(1 + Rf)
    FinalPfValTotalRet = np.prod(1 + totalReturns, axis=0)

    GeomAvgRf = 100 * (FinalPfValRf**(annualizationFactor / nPeriods) - 1)
    GeomAvgTotalReturn = 100 * (FinalPfValTotalRet**(annualizationFactor / nPeriods) - 1)
    GeomAvgXsReturn = GeomAvgTotalReturn - GeomAvgRf

    # ---------- REGRESSION ----------
    alphaArithmetic = np.zeros((1, nAssets))
    alphaPVal = np.zeros((1, nAssets))

    betas = np.zeros((nFactors, nAssets))
    betasPVal = np.zeros((nFactors, nAssets))

    trackingError = np.zeros((1, nAssets))

    x = sm.add_constant(factorXsReturns)

    for asset in range(nAssets):
        y = xsReturns[:, asset]
        model = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': 1})

        # alpha
        alphaArithmetic[0, asset] = 100 * annualizationFactor * model.params[0]
        alphaPVal[0, asset] = model.pvalues[0]

        # betas + p-values
        betas[:, asset] = model.params[1:]
        betasPVal[:, asset] = model.pvalues[1:]

        trackingError[0, asset] = 100 * np.sqrt(annualizationFactor) * np.sqrt(model.mse_resid)

    # ---------- GEOMETRIC ALPHA ----------
    bmRet = factorXsReturns @ betas + Rf
    FinalPfValBm = np.prod(1 + bmRet, axis=0)
    GeomAvgBmReturn = 100 * (FinalPfValBm**(annualizationFactor / nPeriods) - 1)
    alphaGeometric = GeomAvgTotalReturn - GeomAvgBmReturn

    # ---------- OTHER STATS ----------
    xsReturnsPct = xsReturns * 100
    totalReturnsPct = totalReturns * 100

    ArithmAvgTotalReturn = annualizationFactor * np.mean(totalReturnsPct, axis=0)
    ArithmAvgXsReturn = annualizationFactor * np.mean(xsReturnsPct, axis=0)
    StdXsReturns = np.sqrt(annualizationFactor) * np.std(xsReturnsPct, axis=0, ddof=1)

    SharpeArithmetic = ArithmAvgXsReturn / StdXsReturns
    SharpeGeometric = GeomAvgXsReturn / StdXsReturns

    MinXsReturn = np.min(xsReturnsPct, axis=0)
    MaxXsReturn = np.max(xsReturnsPct, axis=0)
    SkewXsReturn = skew(xsReturnsPct, axis=0)
    KurtXsReturn = kurtosis(xsReturnsPct, axis=0)

    # ---------- AUTOCORR ----------
    AC = np.zeros((3, nAssets))
    for asset in range(nAssets):
        s = pd.Series(xsReturnsPct[:, asset])
        AC[0, asset] = s.autocorr(lag=1)
        AC[1, asset] = s.autocorr(lag=2)
        AC[2, asset] = s.autocorr(lag=3)

    # ---------- BUILD TABLE ----------
    rows = [
        'Arithm Avg Total Return', 'Arithm Avg Xs Return',
        'Std Xs Returns', 'Sharpe Arithmetic',
        'Geom Avg Total Return', 'Geom Avg Xs Return',
        'Sharpe Geometric', 'Min Xs Return', 'Max Xs Return',
        'Skewness', 'Excess Kurtosis',
        'Alpha Arithmetic', 'Alpha Arithmetic p-val',
        'Alpha Geometric'
    ]

    # add betas + pvals
    for i in range(nFactors):
        rows.append(f'Beta {i+1}')
        rows.append(f'Beta {i+1} p-val')

    rows += ['Tracking Error', 'Information Ratio', 'AC 1', 'AC 2', 'AC 3']

    # stack values
    data = [
        ArithmAvgTotalReturn, ArithmAvgXsReturn, StdXsReturns, SharpeArithmetic,
        GeomAvgTotalReturn, GeomAvgXsReturn, SharpeGeometric,
        MinXsReturn, MaxXsReturn, SkewXsReturn, KurtXsReturn,
        alphaArithmetic, alphaPVal, alphaGeometric
    ]

    # betas + pvals
    for i in range(nFactors):
        data.append(betas[i, :])
        data.append(betasPVal[i, :])

    InfoRatio = alphaGeometric / trackingError
    data += [trackingError, InfoRatio, AC[0], AC[1], AC[2]]

    df = pd.DataFrame(np.vstack(data), index=rows, columns=strategyNames)

    return df.round(4)


#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------