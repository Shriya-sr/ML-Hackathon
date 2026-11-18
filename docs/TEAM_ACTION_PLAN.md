# PathSnap: Team Action Plan (step by step, per member)

**Roles**
* **Sujal**: owns **Laptop 1**, the backend/API. It holds the full project, model, data and the evaluator trips.
* **Shubashitha**: owns **Laptop 2**, the frontend UI. It only gets the frontend zip.
* **Shriya**: GitHub repository owner, data and features explainer.
* **Sudheeksh**: models, cross-validation and Bayesian optimisation explainer.

Command sheets:
* Laptop 1 → [`LAPTOP1_BACKEND_COMMANDS.md`](LAPTOP1_BACKEND_COMMANDS.md)
* Laptop 2 → [`LAPTOP2_FRONTEND_COMMANDS.md`](LAPTOP2_FRONTEND_COMMANDS.md) (also inside the zip)

---

## Phase A: before the evaluation day

### Sujal (Laptop 1)

1. Confirm the backend starts and passes its self-check. Follow `LAPTOP1_BACKEND_COMMANDS.md`, sections 1–4.
2. Allow port 8000 through the Windows firewall once (section 2, Administrator PowerShell).
3. Send `PathSnap_Laptop2_Frontend.zip` (in `D:\ML hackathon\`) to Shubashitha by pen drive, WhatsApp/Drive or email.
4. Rehearse the two-laptop demo with Shubashitha at least once (Phase B below), on the network you will use on
   the day. Keep a phone hotspot as backup: college Wi-Fi often blocks laptop-to-laptop traffic.
5. Read `reports/viva/TEAM_MEMBER_EXPLANATIONS.md` (your section: inference pipeline, backend, deployment, testing).
6. Optional: add your GitHub username to `TEAM_ROSTER.md`.

### Shubashitha (Laptop 2)

1. Install **Node.js 18 or newer** (https://nodejs.org, LTS installer) if it is not installed. Check with `node --version`.
2. Unzip `PathSnap_Laptop2_Frontend.zip` and follow `LAPTOP2_FRONTEND_COMMANDS.md`, sections 1–3.
   `npm ci` needs internet **once**. If there is no internet on the day, use the no-Node fallback in section 5.
3. Rehearse with Sujal (Phase B).
4. Learn the UI so you can drive it confidently: trip list, layer toggles, **Snap to road**, **Replay trip**,
   clicking a GPS fix ("Why this road?"), the metrics panel and the Change button for the API address.
5. Read your viva section (HMM/Viterbi, Folium maps, UI).

### Shriya

1. **Make the GitHub repository public**: github.com/Shriya-sr/ML-Hackathon → Settings → General → Danger Zone →
   Change visibility → Public. This is required by the submission guidelines.
2. Check that `main` shows the final commit and that the README renders.
3. Add the real GitHub usernames to `TEAM_ROSTER.md`. Edit `TEAM_CONTRIBUTIONS.md` so the split of work matches
   what each member actually does (send the edits to Sujal or commit them yourself).
4. Read your viva section (data, preprocessing, label alignment, candidates, features) and open
   `reports/eda/data_quality_report.md` and the EDA figures.

### Sudheeksh

1. Read your viva section (models, grouped CV, metrics, Bayesian optimisation).
2. Know these files and numbers: `experiments/results/model_summary.csv`, `experiments/optimisation/tuned_vs_untuned.csv`,
   and the model-selection table in the report (§5.2). Be ready to explain why HGB was chosen over the tied Random Forest
   and why tuning barely changed CV Macro-F1.

### Everyone

1. Read `reports/viva/QUICK_REVISION.md` (all the real numbers on one page) and `reports/viva/COMPLETE_VIVA_GUIDE.md`.
2. Go through `reports/viva/QUESTION_BANK.md` (120 questions). Each member answers questions from the
   *other* members' areas too, because evaluators may ask anyone anything.
3. Skim the technical report `reports/technical_report/PathSnap_Technical_Report.pdf`.

---

## Phase B: rehearsal / evaluation day (in this order)

| Step | Who | What |
|---|---|---|
| 1 | Sujal + Shubashitha | Connect both laptops to the **same** Wi-Fi or hotspot |
| 2 | Sujal | If the evaluator trips have been handed over, copy them into `data\raw\evaluator_test\` (Laptop 1 section 3) |
| 3 | Sujal | Start the backend (Laptop 1 section 4) and read out Laptop 1's IP (`ipconfig`) |
| 4 | Shubashitha | Check connectivity to that IP (Laptop 2 section 2) |
| 5 | Shubashitha | Start the UI and set the API address (Laptop 2 section 3) |
| 6 | Shubashitha | Top bar must show **API online**, **EVALUATION** (or DEVELOPMENT if no evaluator trips) and **Trips 10** |
| 7 | Evaluator | Picks any trip; Shubashitha clicks it |
| 8 | Sujal | **Shows the evaluator Laptop 1's console**: a `POST /api/predict/<id> -> 200` line plus the accuracy line appear for every click |
| 9 | Shubashitha | Walks through the map (layers, Snap to road, click a fix) and the metrics panel |
| 10 | Shriya / Sudheeksh | Answer questions on data, features, models, optimisation |

### Screenshots for the report (take them during the rehearsal)

| Who | Screenshot | Save as (on Laptop 1, inside the project) |
|---|---|---|
| Shubashitha | UI with a processed trip (Win + Shift + S) | `reports\figures\ui_trip_<tripid>.png` (e.g. `ui_trip_00000052.png`, `ui_trip_00000009.png`) |
| Sujal | Laptop 1 PowerShell window showing the request log | `reports\figures\api_console_log.png` |

Then Sujal rebuilds the report and commits (Laptop 1 section 6).

---

## Phase C: submission checklist

- [ ] Repository is **public** (Shriya)
- [ ] `TEAM_ROSTER.md` / `TEAM_CONTRIBUTIONS.md` updated with real names, usernames and work split
- [ ] Two-laptop screenshots in `reports/figures/`, report rebuilt, committed and pushed (Sujal)
- [ ] Technical report PDF submitted: `reports/technical_report/PathSnap_Technical_Report.pdf`
- [ ] Every member can explain their part and has gone through the question bank

## If something goes wrong on the day

| Problem | Fix |
|---|---|
| UI says **API offline** | Same Wi-Fi? Correct IP in the UI (Change button)? Backend window still running? Firewall rule added? Try the phone hotspot |
| `Test-NetConnection` fails | Firewall rule (Laptop 1 section 2), or the Wi-Fi is set to *Public*: Settings → Network → Wi-Fi → *Private network* |
| Map background is blank/grey | No internet for map tiles. Everything else still works; untick **Basemap** |
| `npm ci` fails on Laptop 2 (no internet) | Use the no-Node fallback (Laptop 2 section 5) |
| Laptop 2 dies | Run the UI on Laptop 1 too (Laptop 1 section 7) and explain the fallback |
| A trip is slow | The largest trips take 2–3 s; wait for the spinner |
