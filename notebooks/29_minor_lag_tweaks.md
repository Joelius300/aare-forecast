# Last minute lag tweaks

I had noticed a while ago that I never explicitly checked the combination of lag -1,-2 and then higher like -6, -12, etc.
because I did bayesian optimization with optuna and didn't want to try _every_ lag combination because of speed and
overfitting respectively overtuning. But since the model has been struggling on the lowest lags, and we're not using
the full 96h forecast anyway, I thought I'd try that. Should've also implemented exponential lags -1, -2, -4, -8, etc.
As hoped, adding the -2 lag explicitly improves the accuracy of the model (on the validation set) at lower horizons
without significant degradation or instability on the higher lags.

Also, this is a Markdown file because all the commands are just calls to generic scripts, isn't software great?

Trained with

```bash
uv run scripts/train.py lr --targets temp_bern --future tt_bern tt_bern_log tt_bern_cube tt_bern_sqrt tt_bern_ma3 \
  --hparams regularization=none lag_max=24 lag_step=6 extra_lags=[-2]
```

Baseline performance of nowcasting_temp-1.2:

```
uv run scripts/eval.py nowcasting_temp-1.2 --stride 1 --no-store --horizon 1
Evaluation of nowcasting_temp-1.2:
MAE: 0.067 (STD: 0.09) / RMSE: 0.067 (STD: 0.09) / MADPD: 0.067 (STD: 0.09)

uv run scripts/eval.py nowcasting_temp-1.2 --stride 1 --horizon 6 --no-store
Evaluation of nowcasting_temp-1.2:
MAE: 0.188 (STD: 0.21) / RMSE: 0.218 (STD: 0.23) / MADPD: 0.154 (STD: 0.24)

uv run scripts/eval.py nowcasting_temp-1.2 --stride 1 --horizon 12 --no-store
Evaluation of nowcasting_temp-1.2:
MAE: 0.269 (STD: 0.26) / RMSE: 0.315 (STD: 0.29) / MADPD: 0.261 (STD: 0.30)

uv run scripts/eval.py nowcasting_temp-1.2 --stride 1 --horizon 36 --no-store
Evaluation of nowcasting_temp-1.2:
MAE: 0.392 (STD: 0.34) / RMSE: 0.471 (STD: 0.39) / MADPD: 0.423 (STD: 0.38)

uv run scripts/eval.py nowcasting_temp-1.2 --stride 1 --horizon 48 --no-store
Evaluation of nowcasting_temp-1.2:
MAE: 0.434 (STD: 0.37) / RMSE: 0.525 (STD: 0.41) / MADPD: 0.466 (STD: 0.40)

uv run scripts/eval.py nowcasting_temp-1.2 --stride 1 --horizon 72 --no-store
Evaluation of nowcasting_temp-1.2:
MAE: 0.504 (STD: 0.39) / RMSE: 0.611 (STD: 0.43) / MADPD: 0.522 (STD: 0.42)

uv run scripts/eval.py nowcasting_temp-1.2 --stride 1 --horizon 96 --no-store
Evaluation of nowcasting_temp-1.2:
MAE: 0.557 (STD: 0.38) / RMSE: 0.682 (STD: 0.43) / MADPD: 0.566 (STD: 0.43)
```

Compared with the performance of the newly trained LR-dev:

```
uv run scripts/eval.py LR-dev --stride 1 --no-store --horizon 1
Evaluation of LR-dev:
MAE: 0.048 (STD: 0.08) / RMSE: 0.048 (STD: 0.08) / MADPD: 0.048 (STD: 0.08)

uv run scripts/eval.py LR-dev --stride 1 --horizon 6 --no-store
Evaluation of LR-dev:
MAE: 0.166 (STD: 0.19) / RMSE: 0.194 (STD: 0.22) / MADPD: 0.133 (STD: 0.23)

uv run scripts/eval.py LR-dev --stride 1 --no-store --horizon 12
Evaluation of LR-dev:
MAE: 0.250 (STD: 0.25) / RMSE: 0.298 (STD: 0.28) / MADPD: 0.247 (STD: 0.30)

uv run scripts/eval.py LR-dev --stride 1 --no-store --horizon 36
Evaluation of LR-dev:
MAE: 0.383 (STD: 0.34) / RMSE: 0.463 (STD: 0.39) / MADPD: 0.424 (STD: 0.38)

uv run scripts/eval.py LR-dev --stride 1 --no-store --horizon 48
Evaluation of LR-dev:
MAE: 0.430 (STD: 0.37) / RMSE: 0.520 (STD: 0.41) / MADPD: 0.467 (STD: 0.40)

uv run scripts/eval.py LR-dev --stride 1 --no-store --horizon 72
Evaluation of LR-dev:
MAE: 0.501 (STD: 0.38) / RMSE: 0.612 (STD: 0.43) / MADPD: 0.526 (STD: 0.42)

uv run scripts/eval.py LR-dev --stride 1 --no-store --horizon 96
Evaluation of LR-dev:
MAE: 0.558 (STD: 0.38) / RMSE: 0.686 (STD: 0.43) / MADPD: 0.573 (STD: 0.43)
```



<details>
<summary>
I also tried -2 _and_ -3 as extra lags but that performs exactly the same way.
</summary>

```
uv run scripts/train.py lr --targets temp_bern --future tt_bern tt_bern_log tt_bern_cube tt_bern_sqrt tt_bern_ma3 \
  --hparams regularization=none lag_max=24 lag_step=6 extra_lags=[-2,-3]

?
?
?

uv run scripts/eval.py LR-dev --stride 1 --horizon 6 --no-store
Evaluation of LR-dev:
MAE: 0.166 (STD: 0.19) / RMSE: 0.194 (STD: 0.22) / MADPD: 0.133 (STD: 0.23)

uv run scripts/eval.py LR-dev --stride 1 --horizon 12 --no-store
Evaluation of LR-dev:
MAE: 0.250 (STD: 0.25) / RMSE: 0.298 (STD: 0.28) / MADPD: 0.247 (STD: 0.30)

uv run scripts/eval.py LR-dev --stride 1 --horizon 36 --no-store
Evaluation of LR-dev:
MAE: 0.383 (STD: 0.34) / RMSE: 0.463 (STD: 0.39) / MADPD: 0.424 (STD: 0.38)

uv run scripts/eval.py LR-dev --stride 1 --horizon 48 --no-store
Evaluation of LR-dev:
MAE: 0.430 (STD: 0.37) / RMSE: 0.521 (STD: 0.41) / MADPD: 0.467 (STD: 0.40)

uv run scripts/eval.py LR-dev --stride 1 --horizon 72 --no-store
Evaluation of LR-dev:
MAE: 0.501 (STD: 0.38) / RMSE: 0.612 (STD: 0.43) / MADPD: 0.526 (STD: 0.42)

uv run scripts/eval.py LR-dev --stride 1 --horizon 96 --no-store
Evaluation of LR-dev:
MAE: 0.558 (STD: 0.38) / RMSE: 0.686 (STD: 0.43) / MADPD: 0.573 (STD: 0.43)
```
</details>

Finalized the model with

```bash
uv run scripts/train.py lr --targets temp_bern --future tt_bern tt_bern_log tt_bern_cube tt_bern_sqrt tt_bern_ma3 \
  --hparams regularization=none lag_max=24 lag_step=6 extra_lags=[-2] --name nowcasting_temp --version 1.3

uv run scripts/eval.py nowcasting_temp-1.3 --stride 1

uv run scripts/eval.py nowcasting_temp-1.3 --stride 1 --test
```

What's missing is

1. Simulated forecasts to compare visually and enable "vetoing" if something looks really off (I did check test eval tho)
2. Confirmation that the timing assumptions used to determine the forecast schedule (notebook 27) still hold, 
   but in theory they should, unless the focus on the lag 2 of the air temperature has changed dramatically. We should
   monitor this closely, but I think we can be brave for a start.
