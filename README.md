# Alpha Engine Pro

An interactive Streamlit research dashboard for equity market data, return predictions, backtesting, portfolio risk, and model monitoring. The default ticker is Netflix (`NFLX`); the sidebar accepts other Yahoo Finance tickers.

[Open the live dashboard](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app) · [Model results](RESULTS.md) · [Contributing](CONTRIBUTING.md)

> **Research software only.** Predictions, simulated trades, and backtests are experimental. They are not investment advice or a promise of future performance.

## Dashboard

Run the Streamlit app at `app/app.py`. The dashboard includes market charts and technical indicators, next-day return prediction, a chronological holdout backtest, paper trading, sentiment, position risk, feature drift, explainability, AI market narration, architecture, a strategy lab, a research copilot, a signal track record, compliance checks, and observability.

Plotly charts support hover, zoom, pan, legend toggles, and PNG export. Local model and feature-cache files are included, so first launch does not require model training. Market charts try Yahoo Finance and fall back to the included data when live data is unavailable.

## Run from VS Code

Use Python 3.11. Open this repository folder in VS Code, open **Terminal → New Terminal**, and run these commands in PowerShell from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Open <http://localhost:8501>. To launch without activating the environment, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

If PowerShell blocks activation, either use the direct launch command above or allow locally created scripts for your user account with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### If the C: drive is low on space

The XGBoost wheel is large, and pip also needs temporary space while installing it. If C: is nearly full, install Python 3.11 to a drive with free space (for example, `D:\Python311`) using the Python installer, then create the virtual environment and pip temp/cache directories on that drive. Adjust the Python path if you chose another install location:

```powershell
New-Item -ItemType Directory -Force D:\pip-temp, D:\pip-cache, D:\venvs | Out-Null
$env:TEMP = 'D:\pip-temp'
$env:TMP = 'D:\pip-temp'
$env:PIP_CACHE_DIR = 'D:\pip-cache'
& 'D:\Python311\python.exe' -m venv D:\venvs\netflix-stock
& D:\venvs\netflix-stock\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

If an earlier install left a broken `.venv` in the repository, remove that generated environment after switching to the new one with `Remove-Item -Recurse -Force .venv`.

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

The first install can take a few minutes because it includes the tree-model libraries used by the checked-in model. `requirements.txt` contains the dashboard dependencies. AI narration and retrieval features have additional packages; install them only if needed:

```powershell
python -m pip install -r requirements-ai.txt
```

Without those optional packages, the main dashboard still starts; AI features that need them show an unavailable message. Optional API and streaming dependencies are listed separately in `requirements-api.txt` and `requirements-streaming.txt`.

## Deploy or update Streamlit Community Cloud

The Community Cloud app must point to this repository, the branch you want to deploy, and `app/app.py`. Select Python 3.11 in the app's advanced settings when creating it. Cloud installs the root `requirements.txt`; pushes to the configured branch trigger an app update. If the app is configured for `main`, changes must be merged into `main` before they appear in the live app.

For a new app, open [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, then set:

- Repository: `SumedhPatil1507/netflix-stock-prediction`
- Branch: `main`
- Main file path: `app/app.py`
- Python: `3.11`

For an existing app, open **Manage app → Settings** and confirm those values. Community Cloud reads the repository's root `requirements.txt` and `.streamlit/config.toml`. Check **Manage app → Logs** if a build or startup fails; use **Reboot** after correcting a build issue.

### Publish from the VS Code terminal

The repository's `main` branch is protected: GitHub requires a pull request, a passing **Tests** check, and verified commit signatures. Do not push directly to `main`. If SSH commit signing is not configured yet, create an SSH key, add its public key to GitHub under **Settings → SSH and GPG keys** as a signing key, then configure Git once:

```powershell
git config --global gpg.format ssh
git config --global user.signingkey "$env:USERPROFILE\.ssh\id_ed25519.pub"
git config --global commit.gpgsign true
```

Commit and push the current dashboard work as a signed snapshot branch. Use a fresh branch name if `publish/streamlit-refresh` already exists. The source branch below is the current dashboard update branch; substitute your working branch if it has a different name.

```powershell
git fetch origin
git switch -c publish/streamlit-refresh origin/main
git merge --squash publish/dashboard-updates-signed
git add -A
git commit -S -m "Update Streamlit dashboard and documentation"
git push -u origin publish/streamlit-refresh
```

On GitHub, create a pull request from `publish/streamlit-refresh` into `main`. Review the diff and merge after the **Tests** check and signature requirement pass. If Git reports a conflict, resolve it, then stage the resolved files and complete the merge commit before pushing.

After the pull request is merged, Streamlit Community Cloud updates automatically when its configured branch is `main`. If the app tracks another branch, either change the app's branch in its settings or merge the update into that configured branch. Check the Streamlit app logs if the new build does not start.

### Later updates

For changes made after the dashboard work is on `main`, use a new branch based on the latest `origin/main`, then commit and push it:

```powershell
git status
git fetch origin
git switch -c update/my-change origin/main
git add -A
git commit -S -m "Describe the change"
git push -u origin update/my-change
```

Open a pull request from `update/my-change` into `main`, wait for **Tests**, then merge. This is the step that updates Streamlit when its configured branch is `main`.

## Train the stock model

Training is not needed to start the included dashboard. To retrain from the checked-in CSV and regenerate model and output artifacts:

```powershell
python main.py --source csv --ticker NFLX
```

For alternate sources, see `python main.py --help` and the API/data-provider setup in `.env.example`.

## Optional FastAPI service and web UI

The Streamlit dashboard runs independently. The repository also contains a FastAPI service (`api/`) and a Next.js web UI (`web/`). Start the API locally with:

```powershell
python -m pip install -r requirements-api.txt
Copy-Item .env.example .env
# Set a strong API_JWT_SECRET and the required API_USERS_JSON values in .env.
python -m uvicorn api.main:app --reload
```

The API docs are at <http://localhost:8000/docs>. Never commit `.env`, broker credentials, or production secrets. The web UI can be started separately with `cd web; npm ci; npm run dev`; it expects the API to be available and configured.

## Project layout

```text
app/app.py                   Streamlit research dashboard
api/main.py                  FastAPI prediction, risk, execution, and streaming routes
src/                         Data, features, models, backtests, risk, and agent code
data/netflix.csv             Local historical market data
models/model.pkl             Pre-trained model used by the dashboard
outputs/features_cache.parquet  Cached features used by the dashboard and backtest
web/                         Next.js web client
tests/                       Automated tests
```

Run the test suite with `python -m pytest tests/ -q`. See [RESULTS.md](RESULTS.md) for recorded model results and [docs/PERFORMANCE_WHITEPAPER.md](docs/PERFORMANCE_WHITEPAPER.md) for evaluation and observability details.

## License

MIT. See [LICENSE](LICENSE).
