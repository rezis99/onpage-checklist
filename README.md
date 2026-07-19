# TCN Content SOP Validator v4 — Single-File Build

Only 3 files needed in your repo. No folders, no packages, nothing that can
break on upload:

```
app.py               ← everything (1,477 lines, all 7 sections merged)
requirements.txt
.python-version
```

## Why single-file

The previous ModuleNotFoundError happened because the `modules/` folder
never made it into the GitHub repo (GitHub's web upload flattens/skips
folders). This build eliminates the problem entirely — all code lives in
app.py, organized in clearly-marked sections:

1. Utilities & constants
2. Google Sheet data loading
3. Content extraction (live pages + drafts)
4. Embeddings & semantic matching
5. Internal link suggestion engine
6. SOP / Koray / link-health checks
7. Streamlit UI

## Deploy steps

1. In your `onpage-checklist` repo: DELETE the old app.py and the modules/
   folder (if any of it exists).
2. Upload these 3 files to the repo ROOT.
3. Your Streamlit Cloud app will auto-redeploy on the commit (or reboot it
   manually from the dashboard).
4. Python version: the log shows you're already on 3.12 ✅ — no change needed.
5. First run: click "Build / refresh site content index" in the sidebar.
   Building embeddings for ~1,700 pages takes a few minutes the first time,
   then it's cached for 24h.

## Verified before delivery

- Compiles clean on Python 3.12
- Full pipeline executed end-to-end against your real Google Sheet
  (35 anchor rows, 19,991 GSC rows) and real live TCN articles
- Link engine produced correct suggestions (anemia paragraph → low
  hemoglobin article, sim 0.715, correct guide anchor text)
- Org-name violation detection confirmed firing on "Binaytara Foundation"
- Streamlit server booted locally: HTTP 200, zero errors in logs
