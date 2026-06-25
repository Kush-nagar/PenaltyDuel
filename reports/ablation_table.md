# Ablation table - Step 8

Rolling temporal CV. Delta is variant minus full; positive delta_log_loss means the dropped group *helped* (removing it hurt).

| variant | log loss | Brier | ECE | d_log_loss | d_brier | d_ece |
| --- | --- | --- | --- | --- | --- | --- |
| full | 0.5844 | 0.1982 | 0.0564 | +0.0000 | +0.0000 | +0.0000 |
| minus_keeper | 0.5857 | 0.1986 | 0.0365 | +0.0013 | +0.0004 | -0.0199 |
| minus_shooter | 0.5859 | 0.1988 | 0.0613 | +0.0015 | +0.0006 | +0.0049 |
| minus_pressure | 0.5875 | 0.1994 | 0.0446 | +0.0031 | +0.0012 | -0.0117 |
| shrinkage_off | 0.5881 | 0.1996 | 0.0513 | +0.0037 | +0.0014 | -0.0051 |

Reading: a feature group is justified when removing it raises log loss (positive `d_log_loss`). `shrinkage_off` raising log loss confirms shrinkage earns its keep on sparse players.
