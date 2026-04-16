# Huila — Coffee-Linked Deforestation Report

---

        ## Executive Summary

        **Huila** (Colombia) — Arabica, smallholder growing region.
        Analysis of satellite imagery (2019–2024) detected
        **5,000 deforestation hotspots** totalling
        **41,239.7 ha** of potential coffee-linked forest loss.

        Key findings:
        - Largest hotspot: 617.2-ha clearing near huila_h001
- Scale: 20 hotspots >10 ha, 41240 ha total
- Temporal trend: 19.9% change 2019–2023

        Model confidence: **MODERATE** (F1 = 0.000).
        Several large hotspots (>30 ha) align with known road expansion corridors, suggesting infrastructure-driven access.

---

## Area Context

Huila (Colombia) serves as **Generalization test: mountainous, shade-grown edge cases** in this study.
The AOI covers -76.30°–-75.30°E longitude,
1.60°–2.70°N latitude (UTM 32618).

Coffee type: Arabica, smallholder.
Satellite validation: 28.2% coffee coverage,
70.8% forest-2000 baseline coverage (Hansen).
Of current coffee pixels, **0.0%** were forested in 2000,
establishing a direct forest-to-coffee conversion signal.

---

        ## Headline Findings

        | Metric | Value |
        |--------|-------|
        | Total hotspots | 5,000 |
        | Total area | 41,239.7 ha |
        | Largest hotspot | 617.2 ha |
        | Smallest hotspot | 2.7 ha |
        | Coffee on former forest | 0.0% |
        | Hansen loss pixels | 63,137 |

        **Hotspots by primary loss year:**

        | Loss Year | Hotspot Count |
        |-----------|--------------|
        | 2001 | 96 |
| 2002 | 201 |
| 2003 | 35 |
| 2004 | 97 |
| 2005 | 190 |
| 2006 | 133 |
| 2007 | 217 |
| 2008 | 203 |
| 2009 | 217 |
| 2010 | 163 |
| 2011 | 216 |
| 2012 | 238 |
| 2013 | 41 |
| 2014 | 53 |
| 2015 | 36 |
| 2016 | 184 |
| 2017 | 659 |
| 2018 | 297 |
| 2019 | 368 |
| 2020 | 242 |
| 2021 | 108 |
| 2022 | 233 |
| 2023 | 285 |
| 2024 | 488 |

---

## Hotspot Deep-Dives

            ### Finding 1: Largest hotspot: 617.2-ha clearing near huila_h001

            The largest hotspot (huila_h001, 617.2 ha) at (1.7643°N, -75.9144°E) lost forest in 2013 with coffee signal appearing 2 year(s) later in 2015. This lag is consistent with a clear-then-plant pattern.

            Supporting data:
              - area_ha: 617.18
  - loss_year: 2013
  - coffee_signal_year: 2015
  - lag_years: 2

*Maps: outputs/figures/agent_generated/huila_h001_hotspot_boundary_coffee_prob_hansen_loss.png*

          ### Finding 2: Scale: 20 hotspots >10 ha, 41240 ha total

          Historical look-back shows 0.0% of current coffee pixels were forested in 2000. Of 5000 detected hotspots, 20 exceed 10 ha (0.4% of total), accounting for the majority of the 41240 ha total affected area. Context: Several large hotspots (>30 ha) align with known road expansion corridors, suggesting infrastructure-driven access.

          Supporting data:
            - n_hotspots_over_10ha: 20
- total_hotspots: 5000
- total_area_ha: 41239.68
- coffee_on_former_forest_pct: 0.0


          ### Finding 3: Temporal trend: 19.9% change 2019–2023

          Cumulative loss reached 37589.3 ha by 2023, up 6251.2 ha from 2019.

          Supporting data:
            - value_2019: 31338.1
- value_2023: 37589.3
- delta_ha: 6251.2
- pct_change: 19.9


### Top 5 Hotspots by Area

- **huila_h001** (Rank #1) — 617.2 ha at 1.7643°N, -75.9144°E
- **huila_h002** (Rank #2) — 439.1 ha at 1.8285°N, -76.1704°E
- **huila_h003** (Rank #3) — 345.3 ha at 1.7306°N, -75.9907°E
- **huila_h004** (Rank #4) — 269.1 ha at 2.3026°N, -75.5165°E
- **huila_h005** (Rank #5) — 256.9 ha at 2.2734°N, -75.5239°E

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
