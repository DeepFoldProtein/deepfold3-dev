| category | metric | n | AF3 | DF3 r19 @120k | K-Fold | AF3 paper (full set) |
|---|---|---:|---:|---:|---:|---:|
| protein † | lDDT rank-1 | 333 | 0.875 | **0.878** | 0.846 | — |
| protein † | lDDT mean | 333 | 0.875 | **0.879** | 0.836 | 0.88 |
| protein † | lDDT oracle | 333 | 0.882 | **0.885** | 0.856 | — |
| protein † | TM-score rank-1 | 333 | **0.940** | 0.939 | 0.913 | — |
| protein † | GDT-TS rank-1 | 333 | **0.914** | 0.912 | 0.877 | — |
| RNA | lDDT rank-1 | 15 | — | 0.574 | **0.640** | — |
| RNA | lDDT mean | 15 | — | 0.574 | **0.629** | 0.61 |
| RNA | lDDT oracle | 15 | — | 0.596 | **0.658** | — |
| RNA | TM-score rank-1 | 15 | — | 0.275 | **0.365** | — |
| RNA | GDT-TS rank-1 | 15 | — | 0.408 | **0.496** | — |
| DNA | lDDT rank-1 | 14 | — | **0.556** | 0.510 | — |
| DNA | lDDT mean | 14 | — | **0.543** | 0.501 | 0.53 |
| DNA | lDDT oracle | 14 | — | **0.573** | 0.542 | — |
| DNA | TM-score rank-1 | 14 | — | **0.239** | 0.226 | — |
| DNA | GDT-TS rank-1 | 14 | — | **0.499** | 0.417 | — |
| protein–protein | DockQ≥0.23 rank-1 % | 240 | **65.8** | **65.8** | 59.6 | — |
| protein–protein | DockQ≥0.23 mean % | 240 | 65.8 | **66.7** | 60.8 | 72.9 |
| protein–protein | DockQ≥0.23 oracle % | 240 | 70.0 | **73.3** | 66.7 | — |
| protein–protein | DockQ rank-1 | 240 | 0.506 | **0.515** | 0.456 | — |
| protein–protein | iLDDT rank-1 | 209 | 0.535 | **0.548** | 0.481 | — |
| protein–protein | lDDT rank-1 | 209 | 0.788 | **0.791** | 0.757 | — |
| antibody–antigen | DockQ≥0.23 rank-1 % | 138 | 41.3 | 32.6 | **50.0** | — |
| antibody–antigen | DockQ≥0.23 mean % | 138 | 36.2 | 31.9 | **46.4** | 47.9 |
| antibody–antigen | DockQ≥0.23 oracle % | 138 | 57.2 | 49.3 | **65.2** | — |
| antibody–antigen | DockQ rank-1 | 138 | 0.288 | 0.245 | **0.349** | — |
| antibody–antigen | iLDDT rank-1 | 92 | 0.575 | 0.525 | **0.614** | — |
| antibody–antigen | lDDT rank-1 | 92 | 0.855 | 0.855 | **0.861** | — |
| protein–peptide | DockQ≥0.23 rank-1 % | 35 | 71.4 | **74.3** | 71.4 | — |
| protein–peptide | DockQ≥0.23 mean % | 35 | 71.4 | **77.1** | 71.4 | 83.0 |
| protein–peptide | DockQ≥0.23 oracle % | 35 | 80.0 | **91.4** | 85.7 | — |
| protein–peptide | DockQ rank-1 | 35 | 0.515 | **0.546** | 0.462 | — |
| protein–peptide | iLDDT rank-1 | 37 | 0.470 | **0.484** | 0.438 | — |
| protein–peptide | lDDT rank-1 | 37 | **0.849** | 0.848 | 0.838 | — |
| protein–DNA † | iLDDT rank-1 | 170 | 0.597 | **0.658** | 0.657 | — |
| protein–DNA † | lDDT rank-1 | 170 | 0.838 | 0.853 | **0.857** | — |
| protein–RNA † | iLDDT rank-1 | 36 | 0.447 | **0.519** | 0.429 | — |
| protein–RNA † | lDDT rank-1 | 36 | 0.809 | **0.831** | 0.808 | — |

Samples per target (median, on the common set): protein AF3 10/DF3 r19 @120k 10/K-Fold 10; RNA AF3 —/DF3 r19 @120k 10/K-Fold 10; DNA AF3 —/DF3 r19 @120k 10/K-Fold 10; protein–protein AF3 20/DF3 r19 @120k 20/K-Fold 20; antibody–antigen AF3 20/DF3 r19 @120k 20/K-Fold 20; protein–peptide AF3 10/DF3 r19 @120k 10/K-Fold 10; protein–DNA AF3 10/DF3 r19 @120k 10/K-Fold 10; protein–RNA AF3 10/DF3 r19 @120k 10/K-Fold 10

† AF3 row is `af3_ref` (template-free); unmarked interface rows use `af3_ref_tmpl`. Bold = best measured value in the row. Paper column = FoldBench Supp. Table 3, full set, not like-for-like.
