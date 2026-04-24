A few layers of inductive bias you can exploit:

  1. Asset class hierarchy

  - LME base metals (AH=aluminum, CA=copper, PB=lead, ZS=zinc) — close-only, London fix. These are industrial metals, highly cointegrated with each other and with
  FCX/SCCO/TECK/VALE/RIO equities.
  - JPX futures (Tokyo) — gold/platinum/rubber, with Mini/Standard/Rolling-Spot variants on the same underlying. The three gold series should be near-perfectly correlated; same for
  the two platinum series. Treat them as noisy views of one latent price, not independent features.
  - US equities/ETFs — by far the largest block, and itself splittable:

  2. Sector/theme clusters inside US stocks

  - Gold miners: AEM, GOLD, NEM, KGC, FNV, WPM, AG, PAAS, HL + ETFs GDX, GDXJ, NUGT (3x), GLD/IAU (bullion), SLV (silver)
  - Oil & gas: XOM, CVX, COP, BP, SHEL, ENB, CVE, EOG, OXY, DVN, HES, MPC, SLB, HAL, BKR, OIH, XLE, KMI, OKE, WMB, TRGP
  - Industrial metals / mining: FCX, SCCO, TECK, VALE, RIO, CLF, X, STLD, NUE, ALB, LYB, XLB
  - Uranium: CCJ, URA
  - Broad equity ETFs: ACWI, VT, RSP, SPYV, VTV, VYM, VXUS
  - Regional ETFs: EEM, IEMG, VWO, EFA, VEA, VGK, EWJ, EWT, EWY, EWZ, YINN
  - Fixed income: AGG, BND, BNDX, BSV, IEF, SHY, TIP, LQD, JNK, EMB, MBB, IGSB, SPIB, SPTL, VCIT, VCSH, VGIT, VGLT, VGSH
  - Banks/financials: MS, RY, TD, BCS, AMP
  - Machinery: CAT, DE

  3. Redundancy / leverage relationships

  - GLD ≈ IAU (same bullion, different expense ratio) — near-identical returns
  - NUGT ≈ 3× GDX daily return — explicit leverage
  - GDXJ is juniors, should have higher beta to GDX
  - VWO ≈ IEMG ≈ EEM, VEA ≈ EFA, BND ≈ AGG — near-duplicates

  4. OHLC structure

  Don't just average OHLC — the informative derived features are typically:
  - log return: log(close / prev_close)
  - range / volatility: (high - low) / close, or Garman-Klass/Parkinson estimators using all four
  - overnight gap: log(open / prev_close) vs intraday log(close / open)
  - volume × |return| as a dollar-flow proxy

  Averaging OHLC throws away the volatility signal, which is usually one of the better predictors.


  5. The "anchor" set in targets is tiny

  Almost every pair has one leg from a very small pool of anchor assets:

  - LME_AH_Close, LME_CA_Close, LME_PB_Close, LME_ZS_Close (the 4 LME base metals)
  - JPX_Gold_Standard_Futures_Close, JPX_Platinum_Standard_Futures_Close

  That's 6 anchors. The other leg ranges over US stocks/ETFs and FX. So the target matrix is essentially (6 anchors) × (≈70 counterparts), plus a handful of singles, plus a few
  LME-vs-LME and anchor-vs-anchor internal spreads.

  Implication: these 6 anchor prices are the center of gravity of the whole problem. Feature engineering for them (their own lags, their vol, their relationships to each other)
  matters far more than for any individual stock.