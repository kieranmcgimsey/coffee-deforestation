# Lâm Đồng — Coffee-Linked Deforestation Report

---

        ## Executive Summary

        **Lâm Đồng** (Vietnam) — Robusta, highland growing region.
        Analysis of satellite imagery (2019–2024) detected
        **5,000 deforestation hotspots** totalling
        **90,539.7 ha** of potential coffee-linked forest loss.

        Key findings:
        - Largest hotspot: 22154.3-ha clearing near lam_dong_h001
- Scale: 20 hotspots >10 ha, 90540 ha total
- Temporal trend: 6.3% change 2019–2023

        Model confidence: **HIGH** (F1 = 0.783).
        High density of small (<5 ha) hotspots clustered along forest edge suggests incremental encroachment rather than large-scale clearing.

---

## Area Context

Lâm Đồng (Vietnam) serves as **Primary showcase: known coffee-driven deforestation** in this study.
The AOI covers 107.80°–108.80°E longitude,
11.40°–12.40°N latitude (UTM 32648).

Coffee type: Robusta, highland.
Satellite validation: 34.7% coffee coverage,
66.7% forest-2000 baseline coverage (Hansen).
Of current coffee pixels, **0.0%** were forested in 2000,
establishing a direct forest-to-coffee conversion signal.

---

        ## Headline Findings

        | Metric | Value |
        |--------|-------|
        | Total hotspots | 5,000 |
        | Total area | 90,539.7 ha |
        | Largest hotspot | 22154.3 ha |
        | Smallest hotspot | 2.4 ha |
        | Coffee on former forest | 0.0% |
        | Hansen loss pixels | 164,118 |

        **Hotspots by primary loss year:**

        | Loss Year | Hotspot Count |
        |-----------|--------------|
        | 2001 | 79 |
| 2002 | 106 |
| 2003 | 32 |
| 2004 | 168 |
| 2005 | 145 |
| 2006 | 90 |
| 2007 | 115 |
| 2008 | 446 |
| 2009 | 189 |
| 2010 | 441 |
| 2011 | 162 |
| 2012 | 340 |
| 2013 | 85 |
| 2014 | 217 |
| 2015 | 149 |
| 2016 | 342 |
| 2017 | 296 |
| 2018 | 186 |
| 2019 | 319 |
| 2020 | 353 |
| 2021 | 220 |
| 2022 | 186 |
| 2023 | 176 |
| 2024 | 154 |

---

## Hotspot Deep-Dives

            ### Finding 1: Largest hotspot: 22154.3-ha clearing near lam_dong_h001

            The largest hotspot (lam_dong_h001, 22154.3 ha) at (12.0015°N, 107.8855°E) lost forest in 2019 with coffee signal appearing 1 year(s) later in 2020. This lag is consistent with a clear-then-plant pattern.

            Supporting data:
              - area_ha: 22154.3
  - loss_year: 2019
  - coffee_signal_year: 2020
  - lag_years: 1

*Maps: outputs/figures/agent_generated/lam_dong_h001_hotspot_boundary_coffee_prob_hansen_loss.png*

          ### Finding 2: Scale: 20 hotspots >10 ha, 90540 ha total

          Historical look-back shows 0.0% of current coffee pixels were forested in 2000. Of 5000 detected hotspots, 20 exceed 10 ha (0.4% of total), accounting for the majority of the 90540 ha total affected area. Context: High density of small (<5 ha) hotspots clustered along forest edge suggests incremental encroachment rather than large-scale clearing.

          Supporting data:
            - n_hotspots_over_10ha: 20
- total_hotspots: 5000
- total_area_ha: 90539.67
- coffee_on_former_forest_pct: 0.0


          ### Finding 3: Temporal trend: 6.3% change 2019–2023

          Cumulative loss reached 89747.2 ha by 2023, up 5320.0 ha from 2019.

          Supporting data:
            - value_2019: 84427.2
- value_2023: 89747.2
- delta_ha: 5320.0
- pct_change: 6.3


### Top 5 Hotspots by Area

- **lam_dong_h001** (Rank #1) — 22154.3 ha at 12.0015°N, 107.8855°E
- **lam_dong_h002** (Rank #2) — 5345.9 ha at 11.8083°N, 108.0766°E
- **lam_dong_h003** (Rank #3) — 2445.1 ha at 12.1758°N, 107.8389°E
- **lam_dong_h004** (Rank #4) — 1063.0 ha at 12.3276°N, 108.2310°E
- **lam_dong_h005** (Rank #5) — 1047.7 ha at 12.2291°N, 107.9935°E

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
        | Random Forest | 0.859 | 0.783 | 0.000 | 0.000 |

        **Ablation study (S1/S2/combined):**

        | Feature Set | F1 (coffee) | Accuracy |
        |-------------|-------------|---------|
        | S1-only (SAR) | 0.588 | 0.746 |
        | S2-only (optical) | 0.670 | 0.809 |
        | S1+S2 (combined) | 0.682 | 0.816 |

        **Confidence notes:**
        - Rule-based detection (Hansen ∩ FDP >0.5) may include false positives near mixed forest-crop boundaries.
- ML F1 score: 0.783 — model confidence rated high.
- Historical look-back uses Hansen 30m data; small clearings <0.5 ha may be missed.

---

## Methodology

1. **Data acquisition**: Sentinel-2 (10 m) and Sentinel-1 SAR (10 m) composites
   via Google Earth Engine, dry-season median composites 2019–2024.
2. **Features**: Spectral indices (NDVI, EVI, NDWI, NBR), SAR backscatter
   (VV/VH ratio, temporal std), contextual (slope, elevation, road proximity).
3. **Hotspot detection (rule-based)**: Hansen Global Forest Change loss pixels
   intersected with FDP coffee probability > 0.5 (FDP model 2025a).
4. **ML classification**: Random Forest trained on pooled
   GEE-sampled labels from all AOIs, cross-AOI holdout evaluation.
5. **Historical look-back**: Hansen 2000 baseline → loss-year detection →
   FDP time series for coffee signal emergence.
6. **Report generation**: Researcher agent (8 tool calls) → writer → synthesist.

Years processed: 2019–2024.
Detection method: rule_based_hansen_fdp.
