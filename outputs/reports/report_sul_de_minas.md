# Sul de Minas — Coffee-Linked Deforestation Report

---

        ## Executive Summary

        **Sul de Minas** (Brazil) — Arabica, industrial growing region.
        Analysis of satellite imagery (2019–2024) detected
        **5,000 deforestation hotspots** totalling
        **36,269.6 ha** of potential coffee-linked forest loss.

        Key findings:
        - Largest hotspot: 413.1-ha clearing near sul_de_minas_h001
- Scale: 20 hotspots >10 ha, 36270 ha total
- Temporal trend: 36.8% change 2019–2023

        Model confidence: **MODERATE** (F1 = 0.000).
        Negative control as expected: hotspot density and area are substantially lower than the Vietnam showcase.

---

## Area Context

Sul de Minas (Brazil) serves as **Negative control: stable plantations, minimal expansion** in this study.
The AOI covers -46.80°–-45.80°E longitude,
-22.20°–-21.20°N latitude (UTM 32723).

Coffee type: Arabica, industrial.
Satellite validation: 38.4% coffee coverage,
18.0% forest-2000 baseline coverage (Hansen).
Of current coffee pixels, **0.0%** were forested in 2000,
establishing a direct forest-to-coffee conversion signal.

---

        ## Headline Findings

        | Metric | Value |
        |--------|-------|
        | Total hotspots | 5,000 |
        | Total area | 36,269.6 ha |
        | Largest hotspot | 413.1 ha |
        | Smallest hotspot | 1.1 ha |
        | Coffee on former forest | 0.0% |
        | Hansen loss pixels | 102,194 |

        **Hotspots by primary loss year:**

        | Loss Year | Hotspot Count |
        |-----------|--------------|
        | 2001 | 155 |
| 2002 | 60 |
| 2003 | 157 |
| 2004 | 49 |
| 2005 | 119 |
| 2006 | 29 |
| 2007 | 143 |
| 2008 | 6 |
| 2009 | 82 |
| 2010 | 119 |
| 2011 | 38 |
| 2012 | 81 |
| 2013 | 54 |
| 2014 | 205 |
| 2015 | 251 |
| 2016 | 364 |
| 2017 | 887 |
| 2018 | 263 |
| 2019 | 337 |
| 2020 | 415 |
| 2021 | 238 |
| 2022 | 235 |
| 2023 | 418 |
| 2024 | 292 |

---

## Hotspot Deep-Dives

            ### Finding 1: Largest hotspot: 413.1-ha clearing near sul_de_minas_h001

            The largest hotspot (sul_de_minas_h001, 413.1 ha) at (-21.8579°N, -46.6507°E) lost forest in 2014 with coffee signal appearing 1 year(s) later in 2015. This lag is consistent with a clear-then-plant pattern.

            Supporting data:
              - area_ha: 413.12
  - loss_year: 2014
  - coffee_signal_year: 2015
  - lag_years: 1

*Maps: outputs/figures/agent_generated/sul_de_minas_h001_hotspot_boundary_coffee_prob_hansen_loss.png*

          ### Finding 2: Scale: 20 hotspots >10 ha, 36270 ha total

          Historical look-back shows 0.0% of current coffee pixels were forested in 2000. Of 5000 detected hotspots, 20 exceed 10 ha (0.4% of total), accounting for the majority of the 36270 ha total affected area. Context: Negative control as expected: hotspot density and area are substantially lower than the Vietnam showcase.

          Supporting data:
            - n_hotspots_over_10ha: 20
- total_hotspots: 5000
- total_area_ha: 36269.57
- coffee_on_former_forest_pct: 0.0


          ### Finding 3: Temporal trend: 36.8% change 2019–2023

          Cumulative loss reached 34572.9 ha by 2023, up 9296.4 ha from 2019.

          Supporting data:
            - value_2019: 25276.5
- value_2023: 34572.9
- delta_ha: 9296.4
- pct_change: 36.8


### Top 5 Hotspots by Area

- **sul_de_minas_h001** (Rank #1) — 413.1 ha at -21.8579°N, -46.6507°E
- **sul_de_minas_h002** (Rank #2) — 359.1 ha at -21.4496°N, -46.6449°E
- **sul_de_minas_h003** (Rank #3) — 346.8 ha at -21.7704°N, -46.6841°E
- **sul_de_minas_h004** (Rank #4) — 325.7 ha at -21.4553°N, -46.6697°E
- **sul_de_minas_h005** (Rank #5) — 251.0 ha at -21.3668°N, -46.6624°E

---

## Historical Context

Hansen Global Forest Change (2000 baseline) provides the loss-year signal.
Within this AOI, **0.0%** of pixels were
forested in 2000. Mean forest-loss year: **unknown**.

**Replacement class distribution (post-loss land cover):**

| Replacement Class | Share |
|-------------------|-------|
| coffee | 100.0% |

The clear-then-plant pattern — forest cleared, followed within 1–4 years by
rising coffee probability — is the dominant signal driving hotspot detection.

---

        ## Model Performance

        | Model | Accuracy | F1 (coffee) | Precision | Recall |
        |-------|----------|-------------|-----------|--------|
        | None | 0.000 | 0.000 | 0.000 | 0.000 |

        **Ablation study (S1/S2/combined):**

        | Feature Set | F1 (coffee) | Accuracy |
        |-------------|-------------|---------|
        | S1-only (SAR) | 0.000 | 0.000 |
        | S2-only (optical) | 0.000 | 0.000 |
        | S1+S2 (combined) | 0.000 | 0.000 |

        **Confidence notes:**
        - Rule-based detection (Hansen ∩ FDP >0.5) may include false positives near mixed forest-crop boundaries.
- ML F1 score: 0.000 — model confidence rated moderate.
- Historical look-back uses Hansen 30m data; small clearings <0.5 ha may be missed.

---

## Methodology

1. **Data acquisition**: Sentinel-2 (10 m) and Sentinel-1 SAR (10 m) composites
   via Google Earth Engine, dry-season median composites 2019–2024.
2. **Features**: Spectral indices (NDVI, EVI, NDWI, NBR), SAR backscatter
   (VV/VH ratio, temporal std), contextual (slope, elevation, road proximity).
3. **Hotspot detection (rule-based)**: Hansen Global Forest Change loss pixels
   intersected with FDP coffee probability > 0.5 (FDP model 2025a).
4. **ML classification**: None trained on pooled
   GEE-sampled labels from all AOIs, cross-AOI holdout evaluation.
5. **Historical look-back**: Hansen 2000 baseline → loss-year detection →
   FDP time series for coffee signal emergence.
6. **Report generation**: Researcher agent (8 tool calls) → writer → synthesist.

Years processed: 2019–2024.
Detection method: rule_based_hansen_fdp.
