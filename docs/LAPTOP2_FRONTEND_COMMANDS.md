# Laptop 2: Frontend (Evaluator UI) Command Sheet

This laptop runs only the user interface. It holds **no model, no dataset and no reference routes**. All
predictions and scoring happen on Laptop 1 and arrive over the network.

Contents of `PathSnap_Laptop2_Frontend.zip`:

```
PathSnap_Laptop2_Frontend/
├── LAPTOP2_FRONTEND_COMMANDS.md   ← this file
├── lan_check.py                   ← optional connectivity check (needs Python)
└── frontend/
    ├── package.json, package-lock.json, vite.config.ts, tsconfig.json, index.html, .env.example
    ├── src/                       ← React + TypeScript source
    ├── public/
    └── dist/                      ← pre-built UI (used only by the no-Node fallback, section 5)
```

Run everything in **Windows PowerShell**.

> **Always type `npm.cmd`, not `npm`.** On many Windows laptops PowerShell blocks `npm` with
> *"running scripts is disabled on this system"*. `npm.cmd` is the same npm and is not affected.
> (Or run the commands from **Command Prompt** (`cmd`), where plain `npm` also works.)

---

## 0. Requirements (once)

* **Node.js 18 or newer.** Check:

  ```powershell
  node --version
  npm.cmd --version
  ```

  If not installed: download the **LTS** installer from https://nodejs.org, install with defaults, then **open a
  new PowerShell window**.
* Internet access **once** for `npm.cmd ci` (downloads ~190 packages).

## 1. Unzip and install (once, ~1 min)

```powershell
Expand-Archive -Path "$HOME\Downloads\PathSnap_Laptop2_Frontend.zip" -DestinationPath "C:\PathSnap_UI" -Force
cd "C:\PathSnap_UI\PathSnap_Laptop2_Frontend\frontend"
npm.cmd ci
```

(Adjust the first path if the zip is somewhere other than *Downloads*.) Expected: `added 190 packages`.

## 2. Check that Laptop 1 is reachable (every time)

Both laptops must be on the **same Wi-Fi or hotspot**. Get Laptop 1's IPv4 address from the Laptop 1 operator
(below written as `192.168.1.23`; use the real one). The backend must already be running on Laptop 1.

```powershell
Test-NetConnection 192.168.1.23 -Port 8000
```

Expected: `TcpTestSucceeded : True`.

```powershell
Invoke-RestMethod http://192.168.1.23:8000/health
```

Expected: `status : ok` and `model_loaded : True`.

Optional full check, if Python is installed on this laptop:

```powershell
cd "C:\PathSnap_UI\PathSnap_Laptop2_Frontend"
python lan_check.py http://192.168.1.23:8000 --all
```

Expected last line: `ALL CHECKS PASSED`.

## 3. Point the UI at Laptop 1 and start it (every time)

```powershell
cd "C:\PathSnap_UI\PathSnap_Laptop2_Frontend\frontend"
Set-Content -Encoding ascii .env.local "VITE_API_URL=http://192.168.1.23:8000"
npm.cmd run dev
```

Expected: `Local: http://localhost:5173/`. Open **http://localhost:5173** in Chrome or Edge.

If the IP changes later, you don't have to restart: click **Change** in the top-right of the UI, type
`http://<new-ip>:8000` and press **Connect**.

Stop the UI with **Ctrl + C** in the PowerShell window.

## 4. What you should see / demo script

1. Top bar: **API online** (green), mode **EVALUATION** (or **DEVELOPMENT** if Laptop 1 has no evaluator trips yet),
   model `hist_gradient_boosting_tuned`, **Trips 10**.
2. Click any trip in the left list. A spinner appears and the request goes to Laptop 1, whose console prints a
   `POST /api/predict/<id> -> 200` line. Ask the evaluator to look at it.
3. The map shows GPS fixes (blue), the road network (grey), the HMM route (green, arrows = direction), start/end markers.
   * Tick **Classifier-only route** to compare with the route before HMM decoding.
   * **Snap to road** animates every GPS fix onto the road the model matched it to.
   * **Replay trip** drives a marker along the trip.
   * Click any blue dot → **Why this road?** panel lists candidate roads with probabilities and features.
4. Right panel: Top-1 accuracy, Macro-F1, route continuity, candidate miss rate, inference time, and the
   classifier-only vs classifier + HMM comparison. These are computed on Laptop 1; the reference route never
   reaches this laptop.
5. Screenshot for the report: **Win + Shift + S**, save as `ui_trip_<tripid>.png` and send it to the Laptop 1 operator.

## 5. Fallback if `npm.cmd ci` is impossible (no internet / no Node.js)

The zip contains a pre-built UI in `frontend\dist`. Serve it with Python:

```powershell
cd "C:\PathSnap_UI\PathSnap_Laptop2_Frontend\frontend\dist"
python -m http.server 5173
```

Open **http://localhost:5173**. The pre-built UI first looks for the API at `http://localhost:8000`, which is
not where Laptop 1 is, so it shows **API offline**. Click **Change**, enter `http://192.168.1.23:8000` (Laptop 1's
IP) and press **Connect**. The address is remembered in this browser.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `TcpTestSucceeded : False` | Backend not running on Laptop 1, wrong IP, laptops on different networks, or Laptop 1's firewall rule missing. College Wi-Fi often isolates devices; use a phone hotspot for both laptops |
| UI shows **API offline** | Same checks as above, then **Retry connection**; verify the URL in the top bar (Change) |
| `npm` not recognised | Install Node.js LTS and open a **new** PowerShell window |
| *running scripts is disabled on this system* | You typed `npm`: use `npm.cmd` (or use Command Prompt) |
| `npm.cmd ci` errors | Check internet; delete the `node_modules` folder and run `npm.cmd ci` again; or use section 5 |
| Grey map background | No internet for map tiles; untick **Basemap**. Routes and roads still display |
| Port 5173 busy | `npm.cmd run dev -- --port 5174`, then open http://localhost:5174 |
