# Forecasting Air Pollution in Bosnia and Herzegovina
## A technical report on data construction, model comparison, and findings

---

### Executive summary

This project builds a single, gap-aware hourly air-quality panel for Bosnia and Herzegovina
from two separate national monitoring networks, freezes a strict evaluation protocol, and then
uses that protocol to compare four modelling approaches on 24-hour-ahead forecasting of six
pollutants.

The headline finding is that a **time-series foundation model applied zero-shot beats every
alternative tried, by a wide margin**. Chronos-2 reaches a MASE of 0.765 against 1.077 for a
seasonal-persistence baseline — roughly a 29 % reduction in scaled error — without ever being
trained on these series. It wins on all six pollutants, in all four seasons, in both the
heating and non-heating regimes, and at essentially every station.

Two secondary findings matter for how the result should be read. First, **weather covariates
are worth about five times more than spatial covariates**: adding temperature, wind and
humidity improves Chronos by 4.8–5.9 %, while adding neighbouring stations improves it by
2.1 %. Second, the two bespoke deep-learning tracks — a spatial graph neural network and an
xLSTM — did not beat the foundation model. The GNN reached MASE 0.916, better than the
baselines but clearly behind Chronos; the xLSTM was stable on PM2.5 and diverged on everything
else. Both are reported here as they came out.

---

## 1. Data construction

### 1.1 Two networks, one panel

Bosnia and Herzegovina's air quality monitoring is split between two entity-level
hydrometeorological services that publish in different formats, on different schedules, with
different column naming and different station metadata:

- **FHZ** (Federation of Bosnia and Herzegovina) — 16 stations, published as annual Excel
  workbooks with separate files for pollutants, hourly temperature, and wind.
- **RHZ** (Republika Srpska) — 7 stations, published as one workbook per station.

The ingest notebooks (`notebooks/01_ingest/`) handle each source separately because each needs
its own parsing logic — six notebooks covering FHZ hourly pollutants (2021–2025), the FHZ
merge, FHZ temperature, FHZ wind speed, and the two RHZ steps. `notebooks/02_dataset/build_dataset.ipynb`
then maps every source column by name into one schema and produces the merged panel.

The build is not a naive concatenation. It:

- clips out-of-range values that the sources contain (1 wind direction, 6,384 humidity, 5,688
  pressure readings outside physically possible ranges);
- merges 70,127 duplicate station-hour rows;
- reindexes every station to a **complete** hourly range, so that a missing hour exists
  explicitly as `NaN`. This detail matters more than it looks: without it, every downstream
  model silently treats the next available reading as if it were one hour later, quietly
  corrupting every lag feature and every autoregressive context.

The result is `bih_hourly.csv` — 815,136 rows × 20 columns, 23 stations, 12 cities, hourly from
2021-01-01 to 2024-12-31.

### 1.2 Validation

The build was checked against the original source files value by value: 1,639,722 values
compared, with **exactly one** differing by more than 1 %, and per-year, per-pollutant
correlations of 1.00 across the board. This is the kind of check that is easy to skip and
expensive to skip, since a silent unit or column misalignment in a merge of this size would
propagate into every result downstream without ever raising an error.

### 1.3 Cleaning and the `filled` flag

`notebooks/03_eda/eda.ipynb` produces the cleaned export, `bih_hourly_clean.csv` (753,823 rows
× 32 columns). Gaps of up to three hours are linearly interpolated so that models have
continuous context to work with — but **every interpolated point is flagged** in a companion
`<pollutant>_filled` column.

That flag drives the single most important decision in the evaluation protocol: filled values
are permitted as model *input*, and forbidden as scoring *targets*. Without this rule, a
substantial share of the reported accuracy would be models successfully predicting our own
linear interpolation, which tells us nothing about the models.

---

## 2. Exploratory analysis

Two EDA notebooks sit between the dataset build and the modelling. The first
(`notebooks/03_eda/eda.ipynb`) characterises the panel and produces the cleaned export. The
second (`notebooks/03_eda/spatial_eda.ipynb`) is targeted: it exists to answer the specific
questions the GNN needed answered before any architecture work, and it uses the same frozen
artifacts and the same masking rules as scoring, so what it measures is what a model would
actually be trained on.

### 2.1 Coverage, and what the zeros mean

Coverage is deeply uneven, and the first job of the EDA was to establish which zeros are
missing data and which are missing instruments.

| City | PM10 | PM2.5 | SO₂ | NO₂ | O₃ | CO |
|---|---:|---:|---:|---:|---:|---:|
| Brod | 97.0 | 95.1 | 96.2 | 99.0 | 96.9 | 98.3 |
| Banja Luka | 84.5 | 83.4 | 84.7 | 87.0 | 86.9 | 86.0 |
| Trebinje | 71.6 | 69.0 | 75.5 | 82.7 | 75.1 | 87.3 |
| Bihac | 76.8 | 76.7 | 77.0 | 67.9 | 84.0 | 82.4 |
| Livno | 76.3 | 76.2 | 70.5 | 84.0 | 83.4 | 84.5 |
| Sarajevo | 75.9 | 36.7 | 74.3 | 79.1 | 37.1 | 28.3 |
| Doboj | 70.3 | 0.0 | 73.4 | 61.7 | 68.5 | 21.0 |
| Gacko | 81.4 | 0.0 | 94.3 | 79.7 | 0.0 | 0.0 |
| Mostar | 52.8 | 52.8 | 56.3 | 51.6 | 41.2 | 58.0 |
| Prijedor | 49.4 | 47.7 | 71.6 | 65.6 | 51.0 | 63.8 |
| Ugljevik | 47.2 | 0.0 | 65.9 | 96.7 | 0.0 | 0.0 |
| Tuzla | 10.7 | 52.4 | 74.7 | 76.4 | 47.1 | 57.7 |

Three findings came out of this, each of which changed a downstream decision:

**Structural zeros.** Doboj has no PM2.5 instrument; Gacko and Ugljevik have no PM2.5, O₃ or CO
instruments. These are not gaps to impute — the instrument does not exist. Treating them as
missing data would have produced imputed series with no physical basis.

**City-level figures can be artefacts of aggregation.** Tuzla's 10.7 % PM10 coverage looks like
a broken sensor. Drilling to station level shows that no Tuzla station except Trnovac measures
PM10 at all, and Trnovac covers 53.6 % — the city figure is an average over stations measuring
different things. Similarly, Sarajevo's Ambasada station measures only PM2.5, and nothing else.

**Station-level capability had to be made explicit.** This produced `station_groups.json`, which
labels every station-pollutant pair `never` / `sparse` / `ok`. The GNN later used these as its
training partitions instead of running an exhaustive subset search over pollutant combinations,
which is both cheaper and better grounded.

Coverage after short-gap filling: SO₂ 76.0 %, NO₂ 79.2 %, PM10 62.9 %, O₃ 48.6 %, CO 47.4 %,
PM2.5 46.2 %. The EDA also fitted the PM2.5-from-PM10 ratio that became the `ratio_pm25`
baseline, and exported the daily Bijeljina series separately, since it is daily rather than
hourly and cannot join the main panel.

### 2.2 Do stations actually correlate?

Cross-station PM10 correlation, computed on real (non-filled) measurements over the training
period only, ranges from about 0.9 down to slightly negative. The structure is not uniform:

- **Sarajevo basin stations are tightly coupled.** Ilidza–Otoka 0.90, Ilidza–Vogosca 0.87,
  Otoka–Vogosca 0.86, Ilijas–Vogosca 0.82.
- **Isedlo is the exception that proves the rule.** It is a regional background station at
  969 m, above the basin, and it correlates at roughly 0.0 to −0.06 with its own near
  neighbours. Physical proximity is not the same as sharing an airshed.
- **Distant pairs carry little.** Trebinje correlates at 0.00–0.15 with most of the country.

### 2.3 Correlation versus distance

Correlation decays with distance, but not smoothly, and it flattens out well before the network
diameter. The binned trend falls from roughly 0.52 in the closest bin to 0.36 by 50 km, sits
around 0.30–0.37 through the middle distances, and drops to about 0.15 past 250 km.

This is the single most consequential EDA result, because it invalidated the GNN's original
edge construction. The model used k-nearest-neighbours with `k=5`; on partitions of only six to
eight stations, that effectively fully connects the graph, and it gave Trebinje an edge to Banja
Luka — 280 km apart, correlation near zero — purely to fill a slot. It was replaced with a
distance threshold plus a minimum-degree fallback so no station ends up isolated.

The ten closest pairs also show why distance alone is not sufficient. They span correlations
from 0.48 (Otoka–Vijecnica, 5.8 km) to 0.90 (Ilidza–Otoka, 4.7 km). Proximity raises the
expected correlation; it does not determine it.

### 2.4 Does the `same_type` edge earn its place?

The second edge type links stations sharing a classification (Urbana, Industrijska, Urbano
pozadinska, and so on). The test had to control for a confound: same-type stations might simply
be closer together on average, which would make distance the real driver.

| | mean correlation | median | pairs |
|---|---:|---:|---:|
| All pairs, different type | 0.348 | 0.354 | 142 |
| All pairs, same type | 0.384 | 0.361 | 29 |
| Closer half only, different type | 0.385 | 0.399 | 71 |
| Closer half only, same type | 0.509 | 0.487 | 14 |

Across all pairs the effect is marginal (0.384 vs. 0.348). Restricted to the closer half of
distances — where the confound is controlled — the gap widens substantially, to 0.509 vs. 0.385.
Station type carries real information beyond distance, and the `same_type` edge was kept.

### 2.5 Station identity matters

A Kruskal-Wallis test across stations on PM10 level returns H = 68,879, p ≈ 0. Station identity
is a very strong determinant of pollution level — this is the country-scale confirmation of an
earlier Sarajevo-only finding. The practical implication, noted in the notebook, is that
per-station bias terms (a learned per-node embedding) are likely to help a graph model more than
simply adding shared capacity.

### 2.6 Seasonal signature varies enormously by station

Winter-mean minus summer-mean PM10, per station, spans a factor that no shared calendar feature
can represent:

| Station | Δ PM10 (µg/m³) |
|---|---:|
| Ilijas | +64.8 |
| Banja Luka | +59.6 |
| Otoka | +53.2 |
| Vogosca | +47.3 |
| Hadzici | +46.6 |
| Ilidza | +46.1 |
| Prijedor | +45.5 |
| Brod | +30.3 |
| Bjelave | +29.6 |
| Bihac | +26.6 |
| Mostar | +10.8 |
| Vijecnica | +7.2 |
| Gacko | +6.7 |
| Livno | +3.7 |
| Trebinje | −8.2 |
| Isedlo | −15.7 |

The range runs from +64.8 to −15.7. Basin stations in the Sarajevo and Banja Luka areas show
the heating-plus-inversion signature strongly; the coastal-influenced south (Trebinje) and the
high-altitude background station (Isedlo) run *cleaner* in winter than in summer, reversing the
sign. A single shared pair of day-of-year sine and cosine features — which is what the GNN used
— is too coarse to represent a seasonal effect that changes sign across the network. This was
flagged as a limitation of the current model rather than fixed.

Diurnal profiles were examined for the same reason, testing whether traffic-influenced stations
show the expected twin rush-hour peaks while background stations stay flat — which would be
further independent evidence that the `same_type` edge captures something physical.

### 2.7 The premise test

The final section tests the GNN's core premise directly: does a neighbouring station's recent
past help predict this station's future, beyond what the station's own history already provides?
Lagged cross-correlation peaking near lag 0 and decaying quickly indicates real, learnable
spatial structure; a flat profile would mean neighbours add little and a per-station model is
the better fit.

The answer, corroborated later by the ensemble's spatial diagnostic (+2.1 % from neighbour
covariates, section 4.4), is that the structure exists but is modest. Notably, this EDA was
explicitly written to be able to return a negative answer — the summary cell frames each result
as a decision that could go either way, including dropping the `same_type` edge entirely if it
failed the matched-distance test. That framing is why the +2.1 % result later in the project was
reported as marginal rather than presented as a success.

---

## 3. The evaluation protocol

`notebooks/02_dataset/shared_setup.ipynb` freezes the entire experimental design **before** any
model was trained, and writes it to `dataset/shared/`. Three independent model tracks then
import the same module, `bih_shared.py`, for loading, masking and metrics.

This is the structural decision that makes the comparison meaningful. Three tracks developed in
parallel, in separate notebooks, on separate hardware, would otherwise almost certainly have
drifted into slightly different window sets, slightly different masking, or slightly different
MASE denominators — and the resulting table would have been a comparison of protocols rather
than of models.

**Split.** Train 2021-01-01 → 2023-12-31 (560,597 rows). Evaluate on 2024 (193,226 rows). 2025
is excluded entirely, on the explicit grounds that FHZ published no 2025 data and an RHZ-only
year is not comparable to the other three.

**Task.** 24-hour horizon, forecast origins at midnight, 24-hour stride.

**Masking.** Score only where the value is present and `filled` is False. A window requires at
least 12 real target hours to be admitted.

**Metrics.** MAE and RMSE per pollutant in native units. MASE — with the denominator being the
in-context seasonal-naive MAE at lag 24 — as the *only* cross-pollutant summary. The reason is
stated explicitly in the split file: CO is measured in mg/m³ and everything else in µg/m³, so
an average MAE across pollutants is not a quantity that means anything.

**Windows.** 31,688 evaluation windows across 98 station-pollutant series, enumerated in
`eval_windows.csv`: NO₂ 7,055, SO₂ 6,986, PM10 5,001, O₃ 4,390, PM2.5 4,151, CO 4,105.

**Baselines.** Three, computed once and frozen: `persistence_last` (repeat last observation),
`persistence_t24` (repeat the value 24 hours ago), and `ratio_pm25` (PM2.5 from PM10 via the
fitted ratio).

| Pollutant | `persistence_last` | `persistence_t24` | `ratio_pm25` |
|---|---:|---:|---:|
| CO | 1.394 | 1.053 | — |
| NO₂ | 1.512 | 1.040 | — |
| O₃ | 1.403 | 1.050 | — |
| PM10 | 1.162 | 1.077 | — |
| PM2.5 | 1.203 | 1.052 | 1.213 |
| SO₂ | 1.043 | 1.159 | — |

`persistence_t24` is the bar at roughly 1.05–1.08 for most pollutants. Notably, SO₂ is the one
pollutant where repeating the *last* value beats repeating yesterday's value — SO₂ has a weaker
diurnal cycle than the rest.

---

## 4. Models

### 4.1 Chronos and Chronos-2 (`notebooks/04_chronos/`)

Three notebooks, escalating in scope.

**`01_chronos_zeroshot.ipynb`** — univariate zero-shot Chronos on the frozen windows, to
establish whether a foundation model is competitive at all before investing in covariates.

**`02_chronos_all_pollutants.ipynb`** — extends to all six pollutants and runs the key ablation.
On a common 800-window subset:

| Model | MASE | MAE | WQL | vs. t-24 |
|---|---:|---:|---:|---:|
| `chronos_bolt_base` | 0.868 | 7.718 | 0.176 | +25.8 % |
| `chronos2_univariate` | 0.890 | 7.755 | 0.181 | +23.9 % |
| `chronos2_weather` | 0.892 | 6.862 | 0.159 | +23.7 % |
| `persistence_t24` | 1.169 | 10.174 | — | 0 |

This notebook also produced the diagnostic that context quality, not model choice, is the
dominant driver of per-window error:

| Real context coverage | Windows | MASE |
|---|---:|---:|
| > 90 % | 28,279 | 0.747 |
| 75–90 % | 1,823 | 0.922 |
| 50–75 % | 987 | 0.910 |
| 25–50 % | 395 | 0.830 |
| < 25 % | 204 | 1.052 |

A window with a well-populated context is forecast at MASE 0.75; a window with a mostly-empty
context degrades to roughly baseline. This is a data-coverage problem, not a modelling problem,
and it points at where instrumentation investment would pay off.

**`03_chronos2_weather.ipynb`** — the version behind the headline number. Chronos-2 run
multivariate, forecasting PM10 and PM2.5 jointly, with six weather covariates (temperature,
wind speed, humidity, pressure, and wind decomposed into u/v components) drawn from the same
cleaned panel. The weather ablation is decisive:

| Pollutant | with weather | without weather | gain |
|---|---:|---:|---:|
| PM10 | 0.774 | 0.813 | +4.8 % |
| PM2.5 | 0.731 | 0.777 | +5.9 % |

**Final Chronos-2 results across all 31,688 windows:**

| Pollutant | MASE | MAE (native units) | gain vs. t-24 |
|---|---:|---:|---:|
| O₃ | 0.701 | 10.97 µg/m³ | +33.3 % |
| CO | 0.741 | 0.112 mg/m³ | +29.6 % |
| PM2.5 | 0.743 | 9.00 µg/m³ | +29.4 % |
| NO₂ | 0.760 | 5.26 µg/m³ | +27.0 % |
| PM10 | 0.775 | 11.77 µg/m³ | +28.0 % |
| SO₂ | 0.830 | 6.34 µg/m³ | +28.3 % |

By season: spring 0.643, winter 0.729, summer 0.803, autumn 0.890. By station, the gain over
`persistence_t24` ranges from about 27 % (Tuzla-Trnovac) to about 34 % (Sarajevo Bjelave) —
remarkably uniform given how differently covered these stations are.

### 4.2 Spatial GNN (`notebooks/05_gnn/`, with `notebooks/03_eda/spatial_eda.ipynb`)

The GNN track was preceded by a dedicated spatial EDA, and this was the right order to do it in.
That EDA asked whether spatial structure exists at all before a graph model was built to
exploit it, and it answered a specific design question: cross-station PM10 correlation decays
sharply beyond roughly 30–50 km.

That finding directly changed the architecture. The initial edge construction used
k-nearest-neighbours with `k=5`, which — on partitions of only 6–8 stations — was effectively
fully connecting the graph, giving Trebinje an edge to Banja Luka 280 km away purely to fill a
slot. It was replaced with a **distance threshold plus a minimum-degree fallback** so that no
station ends up isolated. A second edge type, `same_type`, links stations sharing a
classification (Urbana, Industrijska, Urbano pozadinska, …), collapsing what the reference
paper treated as separate "same land use" and "same source" relations.

The architecture is ConvLSTM → Dense/ReLU → GAT → Dense/ReLU, with `HeteroConv` summing
per-edge-type contributions at each layer. Context length 168 hours. Trained per partition on
target pollutant plus its strongest correlate rather than the full capability set.

Results, PM10 and PM2.5 only:

| Target | n | MAE | RMSE | MASE |
|---|---:|---:|---:|---:|
| PM2.5 | 4,211 | 13.60 | 25.20 | 1.023 |
| PM10 | 5,613 | 16.07 | 28.29 | 0.970 |

In the final integrated evaluation the GNN scores MASE 0.916 overall — better than every
baseline, clearly behind Chronos-2's 0.765. Its distinguishing strength is **calibration**: a
WQL of 0.0649 against Chronos-2's 0.1445, better by more than a factor of two. If the
downstream use is the probability of exceeding a regulatory threshold rather than a point
forecast, that difference is more valuable than the MASE gap suggests.

### 4.3 xLSTM (`notebooks/06_xlstm/`) — a negative result

Six iterations, kept in full, including the diagnostic notebook. The outcome:

| Pollutant | xLSTM MASE (mean) | median | `persistence_t24` |
|---|---:|---:|---:|
| PM2.5 | **0.845** | 0.652 | 1.049 |
| CO | 119.7 | 0.910 | 1.053 |
| NO₂ | 163.1 | 0.850 | 1.040 |
| O₃ | 1,187.4 | 0.853 | 1.048 |
| SO₂ | 613.0 | 0.652 | 1.159 |
| PM10 | 33,832 | 1.700 | 1.078 |

PM2.5 worked and beat every baseline. Nothing else did. The gap between the means and the
medians is the entire diagnosis: median MASE across all windows is 0.806, which is respectable,
while the mean is 1,765 — a minority of windows diverge completely rather than the model being
uniformly mediocre. The diagnostic notebook traced the worst individual window to a MASE of
272,561, and identified that the MASE denominator itself collapses toward zero on some windows
(three windows had a `mase_scale` of exactly 0), which amplifies any error without bound.

The track is included in the repository because the instability is itself informative, and
because a reader should be able to see what was tried. It is excluded from the results table
because a model that diverges on five of six pollutants is not a model whose average you can
meaningfully quote.

### 4.4 Ensemble (`notebooks/07_ensemble/`)

Two notebooks: a prototype and the full version that produces the final comparison table.

The full notebook opens with a **spatial diagnostic** designed to test the GNN's premise before
committing to the blend. It runs Chronos on 400 Sarajevo windows in three configurations:

| Configuration | MASE | vs. weather-only |
|---|---:|---:|
| weather + neighbours | 0.7768 | +2.1 % |
| weather only | 0.7935 | 0 |
| neighbours only | 0.8060 | −1.6 % |

Median neighbour distance in this set is 7.5 km. The +2.1 % is stated with an explicit reading
guide: above +2 % the spatial signal is real; 0–2 % is marginal; at or below 0, Chronos cannot
use neighbours at all. The result lands just at the boundary of "real but marginal", and the
notebook is careful to note that Chronos seeing neighbours as unstructured extra series is not
the same test as a graph model representing space explicitly.

The blend itself fits per-pollutant, per-regime weights on 2023 and applies them to 2024. The
choice of which Chronos configuration to blend on top of was also made on 2023 — where plain
`chronos2` (0.7294) narrowly beat `chronos2_nbr` (0.7297) — explicitly so that the base was not
selected after seeing the evaluation year.

---

## 5. Final results

All models, all 31,688 frozen 2024 windows, 98 station-pollutant series:

| Model | MASE | MAE | WQL | vs. t-24 |
|---|---:|---:|---:|---:|
| `chronos2_nbr` | **0.7591** | 7.090 | 0.1445 | +29.5 % |
| `ensemble` (Chronos-2 + GNN) | 0.7620 | **7.084** | 0.1446 | +29.2 % |
| `chronos2` | 0.7649 | 7.138 | 0.1449 | +28.9 % |
| `gnn` | 0.9161 | 11.809 | **0.0649** | +14.9 % |
| `diurnal7_reference` | 1.0706 | 10.656 | — | +0.6 % |
| `persistence_t24` | 1.0766 | 10.440 | — | 0 |
| `persistence_last` | 1.2823 | 12.244 | — | −19.1 % |

**By regime.** Heating season is harder than non-heating for every model
(Chronos-2 0.787 vs. 0.743; GNN 0.898 vs. 0.934 — the GNN is the one model that does *not*
follow this pattern, doing relatively better in winter).

**By season.** Spring easiest (0.643), autumn hardest (0.890) for Chronos-2.

**On the 8,473 windows the GNN covers**, where the blend can actually do something, the ordering
tightens: ensemble 0.7473, `chronos2_nbr` 0.7499, `chronos2` 0.7581.

### How to read the top three rows

The three Chronos variants are separated by 0.4–0.8 %. The notebook's own verdict calls
anything under about 1 % "within the noise of a single held-out year", and it is right to.
Two specific pieces of evidence support treating them as a tie:

1. `chronos2_nbr` has the best 2024 MASE but was *worse* on the 2023 tuning year (0.7297 vs.
   0.7294). A sign flip across years on a 0.03 % gap is the signature of noise, not of a real
   effect that would generalise.
2. The `ensemble` is the only one of the three whose advantage is not circular: its weights
   were fitted on 2023 and then beat their own base out-of-sample on 2024, by +0.38 %. It is
   also first on MAE (7.084).

The defensible summary is therefore: **Chronos-2 is the result; the choice among its three
variants is not resolved by this evaluation.** Quoting any one of them as decisively best would
over-claim on a difference the data cannot support.

---

## 6. Findings

**1. Zero-shot foundation models are a strong default for this problem.** Chronos-2 beat a
purpose-built spatial GNN and a purpose-built xLSTM, without training on the target series,
across every pollutant and season. For a domain where per-station training data is patchy and
instrumentation differs station to station, that generalisation is worth a great deal
practically, not just on the leaderboard.

**2. Weather covariates matter about five times more than spatial ones.** +4.8 to +5.9 % from
meteorology versus +2.1 % from neighbouring stations. If effort has to be prioritised, the
meteorological join is the higher-yield one — and it is also far cheaper to build.

**3. The spatial signal is real but small, and it decays fast.** Correlation between stations
falls off sharply past 30–50 km. In a country with 23 stations spread over ~300 km, this means
most station pairs carry little mutual information, which caps how much any graph model can
extract. The GNN's result should be read against that ceiling rather than as a failure of the
architecture.

**4. Data coverage dominates model choice at the window level.** MASE ranges from 0.747 on
well-covered windows to 1.052 on windows with under 25 % real context — a spread larger than
the entire gap between the best and worst *models*. Improving instrument uptime would likely
buy more forecast accuracy than any further modelling work.

**5. Point accuracy and uncertainty calibration come apart.** The GNN is the weakest learned
model on MASE and the strongest on WQL by more than 2×. Which model is "best" therefore depends
on the downstream decision: point forecast, or exceedance probability.

**6. Autumn and the heating season are where the models struggle.** Consistent across every
approach. This is when pollution episodes are most severe in this region, so the errors are
concentrated exactly where forecasts matter most — a natural target for future work.

**7. Negative results were kept.** The xLSTM track did not work and is reported as such. So is
the finding that the ensemble adds only +0.38 %. Both were easy to quietly drop, and the record
is more useful with them in it.

---

## 7. Limitations and future work

- **One held-out year.** All conclusions rest on 2024. Differences under ~1 % cannot be
  resolved with a single year, which is exactly the situation among the three Chronos variants.
  Multi-year rolling-origin evaluation would settle it.
- **The GNN covers only PM10 and PM2.5.** Its overall MASE is therefore not directly comparable
  with Chronos-2's six-pollutant figure, and the ensemble can only differ from plain Chronos-2
  on those two pollutants — for CO, NO₂, O₃ and SO₂ the ensemble's numbers are identical to
  `chronos2` by construction.
- **The GNN was not run to completion across all partitions.** The notebook's own next-steps
  list includes looping over every partition, adding `sparse`-status stations, and a
  season-conditioned breakdown. Its 0.916 should be read as a first full result, not a tuned
  ceiling.
- **The xLSTM instability was diagnosed but not fixed.** The divergence is understood
  (unbounded MASE amplification on windows with a near-zero denominator, plus training
  instability on some series); it was not resolved.
- **No fine-tuning of Chronos was completed.** A LoRA fine-tuning path on 2021–2023 is scaffolded
  in `02_chronos_all_pollutants.ipynb` but was not run to a reported result. Given that the
  zero-shot model already reaches 0.765, this is the most obvious remaining lever.
- **2025 data is excluded**, so the results describe 2024 conditions only.

---

## Appendix: file map

| Path | Purpose |
|---|---|
| `notebooks/01_ingest/fhz_pollutants_hourly.ipynb` | FHZ hourly pollutants, 2021–2025, per-year parsing and merge |
| `notebooks/01_ingest/fhz_merge_all.ipynb` | Combines FHZ pollutant, temperature and wind streams |
| `notebooks/01_ingest/fhz_temperature.ipynb` | Hourly temperature for 9 FHZ cities → single CSV |
| `notebooks/01_ingest/fhz_wind_speed.ipynb` | Wind speed across FHZ cities → single CSV |
| `notebooks/01_ingest/rhz_merge.ipynb` | Merges the 7 per-station RHZ workbooks |
| `notebooks/01_ingest/rhz_to_dataset.ipynb` | RHZ export into the dataset directory |
| `notebooks/02_dataset/build_dataset.ipynb` | Column mapping, range clipping, dedup, gap-free reindex, source validation → `bih_hourly.csv` |
| `notebooks/02_dataset/shared_setup.ipynb` | Freezes split, windows, station geometry, baselines, `bih_shared.py` |
| `notebooks/03_eda/eda.ipynb` | Coverage analysis, instrument-absence findings, PM2.5/PM10 ratio, cleaned export |
| `notebooks/03_eda/spatial_eda.ipynb` | Cross-station correlation, correlation vs. distance, matched-distance `same_type` test, Kruskal-Wallis on station identity, per-station diurnal and seasonal profiles, lagged cross-correlation → graph design |
| `notebooks/04_chronos/01_chronos_zeroshot.ipynb` | Univariate zero-shot Chronos |
| `notebooks/04_chronos/02_chronos_all_pollutants.ipynb` | All six pollutants, model ablation, context-coverage diagnostic |
| `notebooks/04_chronos/03_chronos2_weather.ipynb` | Chronos-2 multivariate + weather covariates — headline model |
| `notebooks/05_gnn/gnn_spatial.ipynb` | Edge construction, ConvLSTM→GAT model, per-partition training and scoring |
| `notebooks/06_xlstm/01…06` | Six xLSTM iterations plus diagnostics (negative result) |
| `notebooks/07_ensemble/01_ensemble_prototype.ipynb` | Spatial diagnostic and first blend |
| `notebooks/07_ensemble/02_ensemble_full.ipynb` | Chronos-2 + GNN blend, final comparison table and verdict |

---

*Evaluation: 31,688 frozen windows, 98 station-pollutant series, calendar year 2024.
Training data: 2021–2023. All figures reproduced from notebook outputs in this repository.*
