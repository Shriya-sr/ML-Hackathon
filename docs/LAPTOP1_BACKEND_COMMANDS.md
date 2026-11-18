# Laptop 1: Backend (Model & API) Command Sheet

Run everything in **Windows PowerShell**. The project lives in `D:\ML hackathon\PATHSNAP` and its Python
environment (`.venv`) is already set up on this laptop. The commands call `.venv\Scripts\python.exe` directly,
so you do **not** need to activate the venv (this avoids PowerShell execution-policy errors).

---

## 1. Go to the project and check it works (once, ~1 min)

```powershell
cd "D:\ML hackathon\PATHSNAP"
.\.venv\Scripts\python.exe -c "import pathsnap, fastapi, sklearn; print('environment OK')"
.\.venv\Scripts\python.exe -m pytest -p no:warnings -q
```

Expected: `environment OK`, then `92 passed`.

## 2. Allow port 8000 through the Windows firewall (once)

Open **PowerShell as Administrator** (Start → type *PowerShell* → right-click → *Run as administrator*):

```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any
```

Close the Administrator window afterwards. If Windows shows a firewall pop-up the first time the API starts,
click **Allow**.

## 3. Put the evaluator trips in place (when you receive them)

Copy the 10 reserved trip folders (each folder holds `<id>.track`, `.nodes`, `.arcs`, `.route`) into
`D:\ML hackathon\PATHSNAP\data\raw\evaluator_test\`. If they arrive as a zip:

```powershell
cd "D:\ML hackathon\PATHSNAP"
Expand-Archive -Path "C:\path\to\evaluator_trips.zip" -DestinationPath "data\raw\evaluator_test" -Force
Get-ChildItem data\raw\evaluator_test
```

The last command must list 10 folders directly (e.g. `data\raw\evaluator_test\00000123\`). If the zip created an
extra top-level folder, move the 10 trip folders up one level.

Without evaluator trips the API runs in **DEVELOPMENT** mode and serves the 10 student-testing trips.

## 4. Start the backend (every time)

```powershell
cd "D:\ML hackathon\PATHSNAP"
ipconfig
```

Note the **IPv4 Address** under *Wireless LAN adapter Wi-Fi* (e.g. `192.168.1.23`). Give this to the Laptop 2
operator.

```powershell
.\.venv\Scripts\python.exe -m backend.main
```

Expected output (keep this window open and visible for the evaluator):

```
| INFO  | pathsnap.api | starting PathSnap API on http://0.0.0.0:8000
| INFO  | pathsnap.api | model loaded: hist_gradient_boosting_tuned
| INFO  | pathsnap.api | mode=EVALUATION partition=evaluator_test trips=10      (DEVELOPMENT/student_test if no evaluator data)
```

Every click on Laptop 2 then prints lines like:

```
| INFO  | pathsnap.api | 192.168.1.42 POST /api/predict/00000052 -> 200 (212 ms)
| INFO  | pathsnap.api | trip=00000052 fixes=262 clf_top1=0.9847 hmm_top1=0.9924 continuity=1.0000 total=0.19s
```

Stop the backend with **Ctrl + C**.

## 5. Self-check from Laptop 1 (optional, in a second PowerShell window)

```powershell
cd "D:\ML hackathon\PATHSNAP"
.\.venv\Scripts\python.exe scripts\lan_check.py http://<YOUR-IPv4>:8000 --all
```

Expected last line: `ALL CHECKS PASSED`. This sends a request to all 10 trips through the network interface.
You can also open `http://localhost:8000/docs` in a browser to see the interactive API documentation.

## 6. After the rehearsal: add the screenshots and rebuild the report

1. Save the screenshots into `D:\ML hackathon\PATHSNAP\reports\figures\`:
   * UI screenshots from Laptop 2 → `ui_trip_00000087.png`, `ui_trip_00000015.png` (overwrite the existing ones)
   * a screenshot of this console with the request log → `api_console_log.png` (overwrite)
2. Rebuild the PDF and commit:

```powershell
cd "D:\ML hackathon\PATHSNAP"
.\.venv\Scripts\python.exe scripts\build_report.py
git add reports
git commit -m "docs: add two-laptop demo screenshots"
git push origin develop
git checkout main
git merge --ff-only develop
git push origin main
git checkout develop
```

## 7. Emergency fallback: run the UI on Laptop 1 as well

Only if Laptop 2 fails. Open a second PowerShell window:

```powershell
cd "D:\ML hackathon\PATHSNAP\frontend"
npm.cmd run dev
```

Then open `http://localhost:5173` in a browser on Laptop 1.

## Quick reference

| Task | Command |
|---|---|
| Start API | `.\.venv\Scripts\python.exe -m backend.main` |
| Find IP | `ipconfig` |
| Health check | `Invoke-RestMethod http://localhost:8000/health` |
| Full check | `.\.venv\Scripts\python.exe scripts\lan_check.py http://<ip>:8000 --all` |
| Run tests | `.\.venv\Scripts\python.exe -m pytest -p no:warnings -q` |
| Use another port | `$env:PATHSNAP_PORT = "8080"` before starting (then also open that port in the firewall and tell Laptop 2) |
