# paper2_agent: LLM agent vs calibrated cost rule (run the whole experiment with one command)

Files
- `run_all.py`       launcher: reads `.env`, runs every combination, collects `summary_all.csv` / `summary_all.md`
- `pilot_agent.py`   one run (called by `run_all.py`); frozen prompt inside, resumable
- `.env.example`     copy to `.env` and fill in (never share `.env`)
- `requirements.txt` `pip install -r requirements.txt`

## On your PC
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`; set `ANTHROPIC_API_KEY`, `MODELS=anthropic:<exact model id>`,
   `RESULTS_DIR` (folder holding `FD001\seed0_preds.npz`, ...).
3. `python run_all.py --dry_run`   (prints the plan and the number of LLM calls, calls nothing)
4. `python run_all.py`

## On Kaggle
Do NOT upload a `.env` that contains your key. Use Kaggle Secrets instead.
Upload this folder + `results/FD001/seed0_preds.npz` as a PRIVATE dataset, turn Internet on, then:

```python
!pip install -q anthropic openai
import os
from kaggle_secrets import UserSecretsClient
os.environ["ANTHROPIC_API_KEY"] = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
```
```python
%%writefile /kaggle/working/.env
MODELS=anthropic:<exact model id>
RESULTS_DIR=/kaggle/input/YOUR-DATASET/results
OUT_DIR=/kaggle/working/agent_runs
DATASETS=FD001
N=50
REPS=3
```
```python
!python /kaggle/input/YOUR-DATASET/paper2_agent/run_all.py --env /kaggle/working/.env --dry_run
!python /kaggle/input/YOUR-DATASET/paper2_agent/run_all.py --env /kaggle/working/.env
```
Download `/kaggle/working/agent_runs` before the session ends.

## Safety
- `MAX_QUERIES` (default 2000): the launcher refuses a larger run unless you add `--yes`.
- A finished query is never repeated: if a run stops, start the same command again.
- The prompt is frozen; its hash is saved in every output line and summary.
