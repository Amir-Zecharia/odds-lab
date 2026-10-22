# Does anything predict EPL outcomes beyond the odds?

Universe: 5150 matches (2012-13 to 2025-26) where Pinnacle pre-close + closing, consensus, and a totals quote all exist; 3250 out-of-sample test matches (2017-18 to 2025-26), expanding-window by season.

| Model | Log loss | vs consensus (95% CI) | vs Pinnacle closing (95% CI) |
|---|---|---|---|
| market consensus (baseline) | 0.9521 | — | — |
| Pinnacle closing (baseline) | 0.9482 | — | — |
| recalibration | 0.9490 | -0.0031 [-0.0060, -0.0004] | +0.0007 [-0.0005, +0.0020] |
| drift | 0.9507 | -0.0014 [-0.0046, +0.0018] | +0.0025 [+0.0007, +0.0043] |
| cross-market | 0.9492 | -0.0030 [-0.0060, +0.0000] | +0.0009 [-0.0007, +0.0026] |
| ensemble | 0.9507 | -0.0014 [-0.0050, +0.0020] | +0.0025 [+0.0005, +0.0046] |

Negative diff = candidate scores better (lower loss). Significant only if the 95% CI excludes zero.

## Verdict

**Negative result.** No candidate — recalibration, closing-line drift, cross-market totals information, or their ensemble — beats both the market consensus and Pinnacle closing at the 95% level on identical test matches. The odds already contain everything these features know.

Moreover, drift, ensemble are **significantly worse** than raw Pinnacle closing (95% CI entirely above zero): fitting extra parameters on information the closing line already prices in doesn't just fail to help — the added estimation noise measurably hurts.

Practical takeaway: recalibration's apparent edge over the consensus is only Pinnacle-closing information flowing through its input — it never beats Pinnacle closing itself. For prediction, use the sharpest quote (PSC, else consensus), de-margined with Shin, and stop there.
