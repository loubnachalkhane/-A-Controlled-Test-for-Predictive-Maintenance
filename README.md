# Does an LLM Agent Add Decision Value over a Calibrated Cost Rule?

Code for the paper "Does an LLM Agent Add Decision Value over a Calibrated Cost Rule? A Controlled Test for Predictive Maintenance".

The paper compares an LLM agent with a calibrated cost rule on a maintenance decision (intervene or wait) and asks whether the agent adds decision value. All arms share the same windows, data splits, horizon and costs. Only the decision maker changes.

## Decision model

- Failure event: `Y = 1{RUL <= H}` for a window with horizon `H`.
- Costs: intervening costs `Cm = 1`; waiting costs `Cf * Y`, with `Cf = 10`, so `tau = Cm / Cf = 0.1`.
- Calibrated failure probability: `F = Phi((H - Q50) / sigma)`, with `sigma = max((Q90 - Q10) / (2 * 1.2816), 1e-6)`.
- Cost rule: intervene if and only if `F > tau`. It is optimal when `F` is calibrated (Theorem 1 of the paper). An error in the agent's implicit probability costs something only when it moves the estimate across `tau` (Theorem 2).

## Arms

| Arm | Agent receives | Status |
|---|---|---|
| Rule | `F` and `tau` (no LLM) | reference |
| Agent-F | `F`, `Q10`, `Q50`, `Q90`, `H`, `Cm`, `Cf` | main, frozen before test results |
| Agent-noF | `Q10`, `Q50`, `Q90`, `H`, `Cm`, `Cf` (no `F`) | exploratory, added after the main results |

Unparseable agent outputs are counted separately and scored as INTERVENE.

## Prompts

The prompts are in `pilot_agent.py`. Their hashes are printed in every run summary.

| Prompt | Hash (sha256, first 16 hex characters) |
|---|---|
| Main (Agent-F) | `4e4190460def0ada` |
| No-F (Agent-noF) | `83f4fcdbdadc64b2` |

## Data

- C-MAPSS FD001 to FD004 (public benchmark), horizons `H = 30, 30, 30, 25`. All test windows are used, one per engine: 100, 259, 100 and 248, so 707 in total.
- N-CMAPSS DS01-005 (public benchmark), `H = 50`. It has 4 test units and 341 test windows; the agent is evaluated on a sample of 300 (80, 80, 70 and 70 per unit).
- Seed 0 only.

Only public benchmark data are used in this paper. The datasets and the per-window predictions of the calibrated quantile model (`Q10`, `Q50`, `Q90`, `F`) are not included in this repository; the predictions are available on request.

## Requirements

- Python 3.10 or later (developed with 3.14)
- An account and API key with an OpenAI-compatible provider. The paper uses Qwen 3.8 27B on Cerebras (`https://api.cerebras.ai/v1`) and gpt-oss-120b on Groq (`https://api.groq.com/openai/v1`).

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill it in. Never commit `.env` or share your API key. `.env` is listed in `.gitignore`.

| Variable | Meaning | Example |
|---|---|---|
| `OPENAI_BASE_URL` | provider endpoint | `https://api.cerebras.ai/v1` |
| `OPENAI_API_KEY` | your key | (secret) |
| `MODELS` | model name | `openai:qwen-3.8-27b` |
| `RESULTS_DIR` | folder with the predictions | `./results` |
| `OUT_DIR` | output folder, one per model | `./agent_runs_cerebras` |
| `DATASETS` | datasets to run | `FD001,FD002,FD003,FD004,NCMAPSS` |
| `SEEDS` | seeds | `0` |
| `N`, `N_NCMAPSS` | windows per dataset (0 = all), N-CMAPSS sample | `0`, `300` |
| `REPS` | repetitions per window | `3` (Qwen main), `1` otherwise |
| `ARMS` | arms | `F` |
| `VARIANTS` | prompt variant | `main` or `noF` |
| `CF_LIST` | cost ratios | `10` |
| `TEMPERATURE`, `MAX_TOKENS` | sampling | `0`, `2000` |
| `WORKERS` | parallel requests | `2` |
| `MAX_QUERIES` | query budget guard | `2000` |

Use a separate `OUT_DIR` for each model, and never mix providers in one result folder.

## Running

Main experiment (Agent-F):

```
python run_all.py --yes
```

For the exploratory run without `F`, set `VARIANTS=noF` and a new `OUT_DIR` in `.env`, then run the same command. A run can be resumed after an interruption (for example a provider's daily token limit) by running it again.

Analysis:

```
python analyze_pooled.py <OUT_DIR> --model qwen
python analyze_pooled.py <OUT_DIR> --variant noF --model qwen
python headroom.py --help
python make_figures_paper2.py --help
```

`analyze_pooled.py` reports, per dataset and pooled: agreement with the rule, costs, the informative subset (`0.02 <= F <= 0.8`), the near-threshold windows (`|F - tau| <= 0.05`), and cluster-bootstrap confidence intervals over engines (10,000 resamples). `headroom.py` reports the reference policies (always intervene, never intervene, oracle), the `Cf` sweep, calibration (ECE with 10 bins) and Clopper-Pearson bounds. `make_figures_paper2.py` writes the paper figures as PDF and PNG.

## Main results (summary)

- When `F` is given, both models match the rule on all 707 C-MAPSS engines and on all 300 sampled N-CMAPSS windows. The exact one-sided 95% upper bound on the disagreement rate is 0.42% (C-MAPSS) and 1.0% (N-CMAPSS).
- Without `F` (exploratory, Qwen only), there are 4 disagreements on C-MAPSS and 5 on N-CMAPSS. All lie within 0.05 of `tau`, and in all of them the agent waits where the rule intervenes. The pooled C-MAPSS cost difference is +0.0085, 95% CI [-0.0085, +0.0368], so equivalence cannot be claimed.
- N-CMAPSS has only 4 test units and weaker calibration (ECE 0.067 against 0.014 on C-MAPSS), so its results are descriptive.

## Limitations

One seed; two models, and one for the run without `F`; the run without `F` is exploratory; at `Cf = 10` the threshold `tau = 0.1` equals the 10% quantile level, so the test without `F` is partly easy; benchmark data only. See the paper for details.

## Repository layout

```
run_all.py              runs the experiments
pilot_agent.py          prompts, agent queries, decision parsing
analyze_pooled.py       pooled analysis and bootstrap
headroom.py             reference policies, Cf sweep, calibration, bounds
make_figures_paper2.py  figures for the paper
.env.example            configuration template (copy to .env)
```

## Citation and license

The citation and the license will be added after publication.
