# Deploy the dashboard to Vercel

[Live QKD dashboard](https://qkd-dashboard-orpin.vercel.app/)

## Automatic deployment from GitHub

1. In the existing Vercel project, open **Settings → Git** and connect the GitHub repository. If GitHub asks which repositories the Vercel app may access, select this repository.
2. Use the repository root and the FastAPI framework. Under **Settings → Environments → Production**, set branch tracking to `main`. The root configuration selects Python 3.13 and checks model compatibility before deployment.
3. Connect Upstash Redis to both Preview and Production and check for `KV_REST_API_URL` and `KV_REST_API_TOKEN` in Vercel's environment settings. The app also accepts the corresponding Upstash REST variables.
4. Verify the first Git deployment in the Vercel activity view. Check `/healthz`, `/api/source/meta`, and blind simulation, reveal, and SHAP for BB84 and both source profiles.

Git builds use the three versioned synthetic inference checkpoints in
`hosted_models/`. Training data and credentials are generated or configured
separately. To refresh the versioned bundle after training, prepare it in a
fresh directory under the repo, review the replacement files, and run the
release check before pushing.

```powershell
.\.venv-phase3\Scripts\python.exe scripts/15_prepare_vercel.py 'results/phase3/runs/<run-id>' --git-assets --output .vercel-qkd/next-hosted-models
# Replace hosted_models/ with the reviewed output, keeping its directory layout.
.\.venv-phase3\Scripts\python.exe scripts/16_check_vercel_release.py
```

## Manual CLI upload

The CLI upload includes the dashboard, FastAPI routes, BB84 checkpoint, both
Phase 3 checkpoints and benchmark tables. Training and dataset replay run
separately. Use this path to deploy checkpoints from a new local run.

From a fresh clone, run these commands in PowerShell with Python 3.13.13.
They install the pinned environment, generate the BB84 dataset and checkpoint,
and train both Phase 3 profiles.

```powershell
python -m venv .venv-phase3
.\.venv-phase3\Scripts\python.exe -m pip install -r requirements-phase3-lock.txt
.\.venv-phase3\Scripts\python.exe scripts/06_generate_multiattack_dataset.py
.\.venv-phase3\Scripts\python.exe scripts/07_train_multiattack_classifiers.py
.\.venv-phase3\Scripts\python.exe scripts/13_source_side_pipeline.py --bounded --seeds 42 43 44 --workers 2
```

If you already have the BB84 checkpoint and a completed bounded or full Phase 3
run, start here. Replace the example path with the completed run path printed by
the pipeline. The default output is `.vercel-qkd`. For a new build, choose a
fresh path with `--output` and link that folder to the same Vercel project.

```powershell
.\.venv-phase3\Scripts\python.exe scripts/15_prepare_vercel.py 'results/phase3/runs/<run-id>'
Set-Location .vercel-qkd
npx --yes vercel login
npx --yes vercel link
```

In Vercel, open the linked project and connect an Upstash Redis database through
Storage or the Marketplace. Ensure the project has `KV_REST_API_URL` and
`KV_REST_API_TOKEN` in both Preview and Production. A manually connected database
can use `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` instead. Used only for
short-lived simulated sessions, Redis lets Reveal and SHAP work across Vercel
instances. Sessions expire after 30 minutes or when the deployment changes.

```powershell
npx --yes vercel deploy
```

Open the preview and check `/healthz`, `/api/source/meta`, a BB84 blind reveal,
and simulation, reveal and explanation with both Source Monitor profiles.
After verification, deploy to the production domain:

```powershell
npx --yes vercel deploy --prod
```

The upload selects Python 3.13 and CPU XGBoost on Linux. A Vercel preview build
is the platform check for native dependencies and function bundle size.
Use the URL printed by the production deployment command.

[Vercel CLI guide](https://vercel.com/docs/projects/deploy-from-cli) ·
[Upstash setup](https://upstash.com/docs/redis/howto/vercelintegration)
