# Two-Laptop Deployment (Windows)

| Laptop | Runs | Holds |
|---|---|---|
| **Laptop 1 – Model & API** | `python -m backend.main` (FastAPI + Uvicorn on `0.0.0.0:8000`) | trained classifier, HMM parameters, road networks, evaluator trips + reference routes (server-side scoring) |
| **Laptop 2 – UI** | `npm.cmd run dev` (Vite dev server, React UI) | nothing sensitive: it only receives predictions and aggregate metrics |

Both laptops must be on the **same network** (same Wi-Fi or a phone hotspot; guest/isolated Wi-Fi
networks often block device-to-device traffic). The map basemap tiles come from OpenStreetMap and need
internet. Without internet the UI still draws the road network, GPS fixes and routes on a dark background.

---

## 0. One-time setup (both laptops)

```powershell
git clone https://github.com/Shriya-sr/ML-Hackathon.git PATHSNAP
cd PATHSNAP
```

**Laptop 1** (Python 3.10+):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
# dataset (see data/README.md)
Copy-Item -Recurse "D:\ML hackathon\student_test" data\raw\student_test
New-Item -ItemType Directory -Force data\raw\evaluator_test
```

The trained artifacts (`artifacts/models/final_classifier.joblib`, `artifacts/metadata/*.json`) are already in
the repository, so Laptop 1 does **not** need to retrain.

**Laptop 2** (Node.js 18+):

```powershell
cd frontend
npm.cmd ci
```

## 1. Find Laptop 1's IP address

On Laptop 1:

```powershell
ipconfig
```

Use the `IPv4 Address` of the Wi-Fi adapter, e.g. `192.168.1.23`. The address changes between networks, so
check it again on the demo day. Nothing in the code hard-codes an IP.

## 2. Allow port 8000 through the Windows firewall (Laptop 1, once)

Open PowerShell **as Administrator** on Laptop 1:

```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```

(If Windows shows a firewall prompt when the API first starts, tick **Private networks** and allow.)
If the Wi-Fi is marked *Public*, either switch it to *Private* in Settings → Network or add `-Profile Any`.

## 3. Put the evaluator trips in place (Laptop 1)

Copy the 10 reserved trip folders into `data\raw\evaluator_test\`. If that folder is empty, the API runs in
**DEVELOPMENT** mode and serves the 10 student-testing trips, with a banner in the UI saying so.

## 4. Start the backend (Laptop 1)

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.main
```

Expected console output:

```
10:15:02 | INFO  | pathsnap.api | model loaded: hist_gradient_boosting_tuned
10:15:02 | INFO  | pathsnap.api | mode=EVALUATION partition=evaluator_test trips=10
10:15:02 | INFO  | pathsnap.api | starting PathSnap API on http://0.0.0.0:8000
```

Optional overrides: `$env:PATHSNAP_PORT=8080`, `$env:PATHSNAP_DATA_DIR="E:\eval_data"`.

## 5. Verify connectivity from Laptop 2

```powershell
Test-NetConnection 192.168.1.23 -Port 8000          # TcpTestSucceeded : True
Invoke-RestMethod http://192.168.1.23:8000/health    # status ok, model_loaded True
```

Or open `http://192.168.1.23:8000/docs` in Laptop 2's browser (interactive API docs).

## 6. Configure and start the UI (Laptop 2)

```powershell
cd frontend
"VITE_API_URL=http://192.168.1.23:8000" | Out-File -Encoding ascii .env.local
npm.cmd run dev
```

Open `http://localhost:5173`. The API address can also be changed at runtime with **Change** in the top bar.
It is saved in that browser only.

## 7. Demonstrate

1. Top bar shows **API online**, the mode (**EVALUATION**), model version and **Trips 10**.
2. Click any of the ten trips. The UI sends `POST /api/predict/<trip_id>` to Laptop 1.
3. **Show the evaluator Laptop 1's console**: each request is logged with the client IP, endpoint, status,
   duration and the server-side accuracy line, e.g.
   ```
   10:16:40 | INFO  | pathsnap.api | 192.168.1.42 POST /api/predict/00000052 -> 200 (912 ms)
   10:16:40 | INFO  | pathsnap.api | trip=00000052 fixes=262 clf_top1=0.9656 hmm_top1=1.0000 continuity=1.0000 total=0.85s
   ```
4. The map shows GPS fixes, road network, the HMM route (green, arrows = direction), classifier-only route
   (toggle), start/end, GPS-gap and chain-break markers. **Snap to road** animates fixes onto matched arcs.
   Clicking a fix shows why its road was chosen.
5. The right panel shows Top-1 accuracy, Macro-F1, route continuity, candidate miss rate, inference time and the
   classifier-only vs HMM comparison, all computed on Laptop 1.
6. Repeat for other trips. All ten are processed live; nothing is precomputed.

## 8. Failure handling you can demonstrate

| Situation | Behaviour |
|---|---|
| Laptop 1 API stopped | UI pill turns **API offline**, trip list shows *Retry connection*; reconnects automatically within 10 s once restarted |
| Unknown / malicious trip id | `404 unknown trip id` (ids are whitelisted against the trip folder) |
| Malformed trip files | `422 trip files are malformed: <file>: <reason>` |
| Model artifact missing/corrupt | API still starts; `/health` = `degraded` with the reason; predictions return `503` |
| Prediction slower than `PATHSNAP_PREDICT_TIMEOUT` | `504` and a Retry button in the UI |
| Trip without reference route | prediction works; metrics panel says *Metrics unavailable* |

## Optional: Docker (Laptop 1)

```powershell
docker compose up --build      # mounts data\raw read-only, serves on :8000
```

Docker is optional. The plain Python start above is the recommended demo path.

## Single-machine check (development)

```powershell
python -m backend.main            # terminal 1
cd frontend; npm.cmd run dev           # terminal 2  -> http://localhost:5173
```
