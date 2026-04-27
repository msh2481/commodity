f"""
Overall pipeline
================

Three sets of features:
1. Input features -- here we add everything that might help, engineered for convenience to the model
2. Main targets -- subset of features that are directly relevant to predicting spread, perhaps (H + L + 2C)/4 for each asset involved in target pairs
3. Aux targets -- being able to predict these would be a good sign, but not immediately relevant to predicting spreads; can include stuff like volatility, or assets not participating in spreads

All these share most of the preprocessing steps:
1. Drop US_Stock_GOLD_*, since its mostly nan; also drop `*_settlement_price` for simplicity, since they are almost equal to `*_Close`
2. Ffill and bfill to fill in missing values
3. Process FX columns using `process_fx` from `utils.py` (going from 38 to 9 cols)
4. log(max(., 1e-9)) everything (except date_id, date, dom and dow)
5. OHLC -> weighted_close = (H + L + 2C) / 4, intraday_volatility = (H - C) * (H - O) + (L - C) * (L - O), overnight_return = (O_t - C_(t-1)), day_return = (C_t - O_t), close_in_range = (C - L) / (H - L)
6. To make things stationary: using rolling window of W=90 days, compute robust z-scores as (x - mu) / sigma, where mu = IQM, sigma = (mean absolute deviation from MAD)

Then, for input features, on top of that, EMA-smooth some of the features, and add some basic indicators, like RSI or Stochastic RSI (based on weighted_close)

Running something on all the features at the same time leads to overfitting.
So I need a pipeline where I can take some small set of features, e.g. only RSI of certain raw feature -- and see how they perform (wrt a single target).

It would be easier to put derived feature computation (especially parametrized parts) into ipynb, and here keep only functions that do this:
then I can compute, e.g. RSI, on whatever feature I want, and with whatever periods I want.

"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import QuantileTransformer
from utils import parse_groups, process_fx, process_ohlc, zscore_robust


def common_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = list(df.filter(regex=r"^US_Stock_GOLD_").columns) + list(
        df.filter(regex=r"^.*_settlement_price").columns
    )
    print("Dropping columns: ", cols_to_drop)
    df = df.drop(columns=cols_to_drop)
    df.columns = [
        col.lower().replace("open_interest", "openinterest") for col in df.columns
    ]
    df = df.ffill().bfill()

    fx_df = df.filter(regex=r"^FX_")
    new_fx_df = process_fx(fx_df)
    df = df.drop(columns=fx_df.columns)
    df = pd.concat([df, new_fx_df], axis=1)

    float_cols = df.select_dtypes(include=[np.floating]).columns
    df[float_cols] = np.log(df[float_cols].clip(lower=1e-9))

    results = [df[["dom", "dow"]].copy()]
    for stem, suffixes in parse_groups(df).items():
        if "close" not in suffixes:
            print(f"Skipping stem {stem} because it doesn't have a close column")
            continue
        selected = df[[f"{stem}_{suffix}" for suffix in suffixes]]
        result = process_ohlc(selected, stem)
        results.append(result)
    df = pd.concat(results, axis=1)
    return df


def get_datasets():
    X_orig = (
        pd.read_csv("data/dated_train.csv").drop(columns=["date"]).set_index("date_id")
    )
    X_train = common_preprocessing(X_orig)
    Y_train = pd.read_csv("data/train_labels.csv")
    Y_train = Y_train.ffill().bfill()
    return X_train, Y_train


def rsi(df: pd.DataFrame, col: str, span: int = 14) -> pd.DataFrame:
    close = df[col + "_weighted_close"]
    diffs = close.diff()
    ema_num = diffs.ewm(span=span, adjust=False).mean()
    ema_den = diffs.abs().ewm(span=span, adjust=False).mean()
    result = ema_num / ema_den
    result[:span] = 0
    return result


def emadiff(df: pd.DataFrame, col: str, span: int = 14) -> pd.DataFrame:
    return df[col].diff().fillna(0).ewm(span=span, adjust=False).mean()


risk_on = [
    "us_stock_enb_adj",
    "us_stock_oke_adj",
    "us_stock_trgp_adj",
    "us_stock_kmi_adj",
    "us_stock_wmb_adj",
    "us_stock_cop_adj",
    "us_stock_dvn_adj",
    "us_stock_eog_adj",
    "us_stock_hes_adj",
    "us_stock_cve_adj",
    "us_stock_oxy_adj",
    "us_stock_xle_adj",
    "us_stock_slb_adj",
    "us_stock_hal_adj",
    "us_stock_oih_adj",
    "us_stock_bp_adj",
    "us_stock_shel_adj",
    "us_stock_mpc_adj",
    "us_stock_bkr_adj",
    "us_stock_cvx_adj",
    "us_stock_xom_adj",
    "us_stock_acwi_adj",
    "us_stock_vt_adj",
    "us_stock_vgk_adj",
    "us_stock_vxus_adj",
    "us_stock_efa_adj",
    "us_stock_vea_adj",
    "us_stock_xlb_adj",
    "us_stock_rsp_adj",
    "us_stock_spyv_adj",
    "us_stock_vtv_adj",
    "us_stock_vym_adj",
    "us_stock_nue_adj",
    "us_stock_stld_adj",
    "us_stock_clf_adj",
    "us_stock_x_adj",
    "us_stock_lyb_adj",
    "us_stock_cat_adj",
    "us_stock_de_adj",
    "us_stock_ry_adj",
    "us_stock_td_adj",
    "us_stock_bcs_adj",
    "us_stock_amp_adj",
    "us_stock_ms_adj",
    "us_stock_emb_adj",
    "us_stock_jnk_adj",
    "us_stock_ccj_adj",
    "us_stock_ura_adj",
    "us_stock_alb_adj",
    "us_stock_yinn_adj",
    "us_stock_rio_adj",
    "us_stock_vale_adj",
    "us_stock_teck_adj",
    "us_stock_fcx_adj",
    "us_stock_scco_adj",
    "us_stock_iemg_adj",
    "us_stock_eem_adj",
    "us_stock_vwo_adj",
    "us_stock_ewz_adj",
    "us_stock_ewj_adj",
    "us_stock_ewt_adj",
    "us_stock_ewy_adj",
]

rates = [
    "us_stock_shy_adj",
    "us_stock_vgsh_adj",
    "us_stock_sptl_adj",
    "us_stock_vglt_adj",
    "us_stock_ief_adj",
    "us_stock_vgit_adj",
    "us_stock_igsb_adj",
    "us_stock_vcsh_adj",
    "us_stock_spib_adj",
    "us_stock_lqd_adj",
    "us_stock_vcit_adj",
    "us_stock_tip_adj",
    "us_stock_bndx_adj",
    "us_stock_bsv_adj",
    "us_stock_mbb_adj",
    "us_stock_agg_adj",
    "us_stock_bnd_adj",
]

precious_metals = [
    "us_stock_gld_adj",
    "us_stock_iau_adj",
    "us_stock_hl_adj",
    "us_stock_slv_adj",
    "us_stock_ag_adj",
    "us_stock_paas_adj",
    "us_stock_fnv_adj",
    "us_stock_nem_adj",
    "us_stock_wpm_adj",
    "us_stock_aem_adj",
    "us_stock_kgc_adj",
    "us_stock_gdxj_adj",
    "us_stock_gdx_adj",
    "us_stock_nugt_adj",
]

lme_base_metals = [
    "lme_ca",  # copper
    "lme_pb",  # lead
    "lme_ah",  # aluminum
    "lme_zs",  # zinc
]

jpx_futures = [
    "jpx_rss3_rubber_futures",
    "jpx_platinum_mini_futures",
    "jpx_platinum_standard_futures",
    "jpx_gold_rolling-spot_futures",
    "jpx_gold_mini_futures",
    "jpx_gold_standard_futures",
]
