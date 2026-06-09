"""
main.py — CardioTracker ML Service v2.2
FastAPI entry point — premium dark medical UI
"""

import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "services", "routers", "middleware"):
    p = os.path.join(BASE_DIR, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

# imports жолына:
from routers import risk, alerts, summary, dynamics, import_data, training_data, explorer, ml_validator



_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[CardioTracker ML] ✅ v2.2.0 — online")
    yield
    print("[CardioTracker ML] 🛑 offline")


app = FastAPI(
    title="CardioTracker ML",
    description=(
        "## CardioTracker ML — ХСН мониторинг жүйесі\n\n"
        "Кардиохирургиялық орталыққа арналған интеллектуалды риск-анализ сервисі.\n\n"
        "**Версия:** 2.2.0 | **Тараз, Қазақстан** | **2026**"
    ),
    version="2.2.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(risk.router,           prefix="/ml", tags=["Классификация рисков"])
app.include_router(alerts.router,         prefix="/ml", tags=["Алерты"])
app.include_router(summary.router,        prefix="/ml", tags=["AI Заключение"])
app.include_router(dynamics.router,       prefix="/ml", tags=["Динамика"])
app.include_router(import_data.router,    prefix="/ml", tags=["Импорт данных"])
app.include_router(training_data.router,  prefix="/ml", tags=["Training Data"])
app.include_router(explorer.router,        tags=["Explorer"])
app.include_router(ml_validator.router, prefix="/ml", tags=["ML Validator"])


# ── Custom Swagger UI ────────────────────────────────────────────
@app.get("/docs", include_in_schema=False)
async def custom_swagger():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="CardioTracker ML — API",
        swagger_favicon_url="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>🫀</text></svg>",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        init_oauth=None,
        oauth2_redirect_url=None,
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "none",
            "filter": True,
            "showExtensions": True,
            "syntaxHighlight.theme": "monokai",
            "tryItOutEnabled": True,
        },
    )


# ── Басты бет ────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CardioTracker ML</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg:     #080b0f;
  --s1:     #0d1117;
  --s2:     #111820;
  --border: rgba(255,255,255,.06);
  --text:   #e8edf3;
  --muted:  #4a5568;
  --dim:    #2d3748;
  --red:    #ff3b3b;
  --red2:   #ff6b6b;
  --orange: #ff8c42;
  --yellow: #ffd166;
  --green:  #06d6a0;
  --blue:   #4cc9f0;
  --mono:   'Space Mono', monospace;
  --sans:   'Syne', sans-serif;
}

html{scroll-behavior:smooth}
body{
  font-family:var(--sans);
  background:var(--bg);
  color:var(--text);
  min-height:100vh;
  overflow-x:hidden;
}

/* ── GRAIN OVERLAY ── */
body::before{
  content:'';
  position:fixed;inset:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
  pointer-events:none;z-index:9999;opacity:.4;
}

/* ── HEARTBEAT LINE ── */
.hb-wrap{position:fixed;top:0;left:0;right:0;height:2px;z-index:100;overflow:hidden}
.hb-line{
  position:absolute;top:0;left:-100%;width:100%;height:100%;
  background:linear-gradient(90deg,transparent 0%,var(--red) 50%,transparent 100%);
  animation:hbscan 3s ease-in-out infinite;
}
@keyframes hbscan{0%{left:-100%}100%{left:100%}}

/* ── NAV ── */
nav{
  position:sticky;top:0;z-index:50;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 40px;height:60px;
  background:rgba(8,11,15,.85);
  backdrop-filter:blur(20px);
  border-bottom:1px solid var(--border);
}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{
  width:32px;height:32px;
  background:var(--red);
  border-radius:8px;
  display:grid;place-items:center;
  font-size:18px;
  box-shadow:0 0 20px rgba(255,59,59,.4);
  animation:pulse 2s ease-in-out infinite;
}
@keyframes pulse{0%,100%{box-shadow:0 0 20px rgba(255,59,59,.4)}50%{box-shadow:0 0 35px rgba(255,59,59,.7)}}
.logo-text{font-weight:800;font-size:15px;letter-spacing:-.01em}
.logo-text span{color:var(--red)}
.nav-right{display:flex;align-items:center;gap:24px}
.status-dot{
  display:flex;align-items:center;gap:7px;
  font-family:var(--mono);font-size:11px;color:var(--green);
}
.status-dot::before{
  content:'';width:7px;height:7px;border-radius:50%;
  background:var(--green);
  box-shadow:0 0 8px var(--green);
  animation:blink 1.5s ease-in-out infinite;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.nav-link{
  font-family:var(--mono);font-size:11px;
  color:var(--muted);text-decoration:none;
  text-transform:uppercase;letter-spacing:.1em;
  transition:color .2s;
}
.nav-link:hover{color:var(--text)}
.btn-docs{
  font-family:var(--mono);font-size:11px;
  padding:8px 18px;border-radius:6px;
  background:var(--red);color:#fff;
  text-decoration:none;font-weight:700;
  letter-spacing:.05em;text-transform:uppercase;
  transition:all .2s;
  box-shadow:0 0 20px rgba(255,59,59,.3);
}
.btn-docs:hover{background:var(--red2);box-shadow:0 0 30px rgba(255,59,59,.5);transform:translateY(-1px)}

/* ── HERO ── */
.hero{
  position:relative;
  padding:120px 40px 100px;
  text-align:center;
  overflow:hidden;
}
.hero-bg{
  position:absolute;inset:0;
  background:
    radial-gradient(ellipse 60% 50% at 50% 0%, rgba(255,59,59,.08) 0%, transparent 70%),
    radial-gradient(ellipse 40% 30% at 20% 80%, rgba(76,201,240,.05) 0%, transparent 60%),
    radial-gradient(ellipse 30% 40% at 80% 60%, rgba(6,214,160,.04) 0%, transparent 60%);
  pointer-events:none;
}

/* ECG SVG */
.ecg-wrap{
  position:absolute;bottom:0;left:0;right:0;height:80px;
  opacity:.15;overflow:hidden;
}
.ecg-path{
  stroke:var(--red);stroke-width:1.5;fill:none;
  stroke-dasharray:800;stroke-dashoffset:800;
  animation:drawecg 2.5s ease forwards .5s;
}
@keyframes drawecg{to{stroke-dashoffset:0}}

.hero-tag{
  display:inline-flex;align-items:center;gap:8px;
  font-family:var(--mono);font-size:11px;
  color:var(--red);letter-spacing:.15em;text-transform:uppercase;
  padding:6px 14px;border:1px solid rgba(255,59,59,.3);
  border-radius:100px;margin-bottom:32px;
  background:rgba(255,59,59,.05);
}
h1{
  font-size:clamp(48px,8vw,88px);
  font-weight:800;line-height:.95;
  letter-spacing:-.04em;
  margin-bottom:8px;
}
h1 .accent{
  color:transparent;
  -webkit-text-stroke:1px rgba(255,59,59,.6);
}
h1 .solid{color:var(--text)}
.hero-sub{
  font-size:16px;color:var(--muted);
  max-width:500px;margin:24px auto 48px;
  font-weight:400;line-height:1.7;
}
.hero-cta{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
.btn-primary{
  display:inline-flex;align-items:center;gap:8px;
  padding:14px 32px;border-radius:8px;
  background:var(--red);color:#fff;
  font-family:var(--mono);font-size:12px;font-weight:700;
  text-decoration:none;letter-spacing:.05em;text-transform:uppercase;
  transition:all .25s;
  box-shadow:0 4px 30px rgba(255,59,59,.35);
}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 40px rgba(255,59,59,.5)}
.btn-ghost{
  display:inline-flex;align-items:center;gap:8px;
  padding:14px 32px;border-radius:8px;
  background:transparent;color:var(--text);
  font-family:var(--mono);font-size:12px;font-weight:700;
  text-decoration:none;letter-spacing:.05em;text-transform:uppercase;
  border:1px solid var(--border);
  transition:all .25s;
}
.btn-ghost:hover{border-color:rgba(255,255,255,.2);background:rgba(255,255,255,.04);transform:translateY(-2px)}

/* ── STATS ── */
.stats{
  display:flex;justify-content:center;gap:0;
  padding:60px 40px;
  border-top:1px solid var(--border);
  border-bottom:1px solid var(--border);
  background:var(--s1);
}
.stat{
  flex:1;max-width:200px;
  text-align:center;
  padding:0 40px;
  position:relative;
}
.stat+.stat::before{
  content:'';position:absolute;left:0;top:50%;transform:translateY(-50%);
  width:1px;height:40px;background:var(--border);
}
.stat-val{
  font-family:var(--mono);font-size:36px;font-weight:700;
  color:var(--text);letter-spacing:-.02em;
  margin-bottom:6px;
}
.stat-val.red{color:var(--red)}
.stat-val.green{color:var(--green)}
.stat-val.blue{color:var(--blue)}
.stat-val.orange{color:var(--orange)}
.stat-label{font-size:11px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}

/* ── ENDPOINTS GRID ── */
.section{padding:80px 40px;max-width:1100px;margin:0 auto}
.section-head{
  display:flex;align-items:baseline;gap:16px;
  margin-bottom:40px;
}
.section-label{
  font-family:var(--mono);font-size:10px;
  color:var(--muted);letter-spacing:.15em;text-transform:uppercase;
}
.section-title{font-size:28px;font-weight:700;letter-spacing:-.02em}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:1px;background:var(--border)}
.card{
  background:var(--s1);
  padding:28px;
  transition:background .2s;
  cursor:default;
  position:relative;
  overflow:hidden;
}
.card::before{
  content:'';
  position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent,var(--red)),transparent);
  opacity:0;transition:opacity .3s;
}
.card:hover{background:var(--s2)}
.card:hover::before{opacity:1}
.card-top{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.method{
  font-family:var(--mono);font-size:9px;font-weight:700;
  padding:3px 8px;border-radius:4px;letter-spacing:.08em;
}
.method.post{background:rgba(255,59,59,.15);color:var(--red)}
.method.get{background:rgba(6,214,160,.15);color:var(--green)}
.path{font-family:var(--mono);font-size:12px;color:var(--text)}
.card-desc{font-size:13px;color:var(--muted);line-height:1.6;margin-bottom:16px}
.card-tags{display:flex;gap:6px;flex-wrap:wrap}
.tag{
  font-family:var(--mono);font-size:9px;
  padding:3px 8px;border-radius:4px;
  background:rgba(255,255,255,.04);
  color:var(--dim);letter-spacing:.05em;
  border:1px solid var(--border);
}

/* ── RISK BANDS ── */
.risk-section{
  padding:60px 40px;
  background:var(--s1);
  border-top:1px solid var(--border);
  border-bottom:1px solid var(--border);
}
.risk-inner{max-width:1100px;margin:0 auto}
.risk-grid{display:flex;gap:1px;background:var(--border);border-radius:12px;overflow:hidden;margin-top:32px}
.risk-band{
  flex:1;padding:24px 20px;
  display:flex;flex-direction:column;gap:8px;
  transition:flex .3s ease;
  cursor:default;
}
.risk-band:hover{flex:2}
.risk-band.a{background:rgba(6,214,160,.08)}
.risk-band.b{background:rgba(76,201,240,.08)}
.risk-band.c{background:rgba(255,209,102,.08)}
.risk-band.cd{background:rgba(255,140,66,.08)}
.risk-band.d{background:rgba(255,59,59,.08)}
.risk-color{width:10px;height:10px;border-radius:50%}
.risk-band.a .risk-color{background:var(--green)}
.risk-band.b .risk-color{background:var(--blue)}
.risk-band.c .risk-color{background:var(--yellow)}
.risk-band.cd .risk-color{background:var(--orange)}
.risk-band.d .risk-color{background:var(--red);box-shadow:0 0 10px var(--red)}
.risk-name{font-size:13px;font-weight:600}
.risk-desc{font-size:11px;color:var(--muted);line-height:1.5;opacity:0;transition:opacity .3s}
.risk-band:hover .risk-desc{opacity:1}

/* ── TERMINAL ── */
.terminal-section{padding:80px 40px;max-width:1100px;margin:0 auto}
.terminal{
  background:var(--s2);
  border:1px solid var(--border);
  border-radius:12px;
  overflow:hidden;
}
.term-head{
  display:flex;align-items:center;gap:8px;
  padding:14px 20px;
  border-bottom:1px solid var(--border);
  background:rgba(255,255,255,.02);
}
.term-dot{width:10px;height:10px;border-radius:50%}
.term-dot.r{background:#ff5f57}
.term-dot.y{background:#ffbd2e}
.term-dot.g{background:#28c941}
.term-title{font-family:var(--mono);font-size:11px;color:var(--muted);margin-left:8px;letter-spacing:.05em}
.term-body{padding:24px;font-family:var(--mono);font-size:13px;line-height:2}
.term-line{display:flex;gap:12px;opacity:0;animation:fadein .3s ease forwards}
.term-line:nth-child(1){animation-delay:.1s}
.term-line:nth-child(2){animation-delay:.4s}
.term-line:nth-child(3){animation-delay:.7s}
.term-line:nth-child(4){animation-delay:1s}
.term-line:nth-child(5){animation-delay:1.3s}
.term-line:nth-child(6){animation-delay:1.6s}
.term-line:nth-child(7){animation-delay:1.9s}
@keyframes fadein{to{opacity:1}}
.t-prompt{color:var(--red)}
.t-cmd{color:var(--text)}
.t-out{color:var(--green)}
.t-dim{color:var(--muted)}
.t-yellow{color:var(--yellow)}
.t-blue{color:var(--blue)}

/* ── FOOTER ── */
footer{
  border-top:1px solid var(--border);
  padding:40px;
  display:flex;align-items:center;justify-content:space-between;
  flex-wrap:wrap;gap:16px;
}
.footer-left{font-family:var(--mono);font-size:11px;color:var(--muted)}
.footer-left span{color:var(--text)}
.footer-right{display:flex;gap:24px}
.footer-link{font-family:var(--mono);font-size:10px;color:var(--muted);text-decoration:none;letter-spacing:.08em;text-transform:uppercase;transition:color .2s}
.footer-link:hover{color:var(--text)}

/* ── VERSION BADGE ── */
.version{
  position:fixed;bottom:24px;right:24px;
  font-family:var(--mono);font-size:10px;
  color:var(--muted);
  padding:6px 12px;border-radius:6px;
  background:rgba(13,17,23,.9);
  border:1px solid var(--border);
  backdrop-filter:blur(10px);
  letter-spacing:.08em;
}

/* scrollbar */
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--dim);border-radius:2px}
</style>
</head>
<body>

<div class="hb-wrap"><div class="hb-line"></div></div>

<nav>
  <div class="logo">
    <div class="logo-icon">🫀</div>
    <div class="logo-text">Cardio<span>Tracker</span> <span style="color:var(--muted);font-weight:400">/ ML</span></div>
  </div>
  <div class="nav-right">
    <div class="status-dot">Online</div>
    <a href="#endpoints" class="nav-link">API</a>
    <a href="#risk" class="nav-link">Риск</a>
    <a href="/explorer" class="btn-docs" style="background:var(--blue)">Explorer →</a><a href="/docs" class="btn-docs">Swagger →</a>
  </div>
</nav>

<!-- HERO -->
<section class="hero">
  <div class="hero-bg"></div>
  <div class="hero-tag">🫀 ХСН Мониторинг · Тараз · 2026</div>
  <h1>
    <div class="solid">Cardio</div>
    <div class="accent">Tracker</div>
  </h1>
  <p class="hero-sub">Интеллектуалды риск-анализ сервисі. 500 пациент · 5 поликлиника · ACC/AHA классификация.</p>
  <div class="hero-cta">
    <a href="/docs" class="btn-primary">
      <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
      Swagger UI
    </a>
    <a href="/health" class="btn-ghost">
      <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      Health Check
    </a>
  </div>

  <!-- ECG SVG -->
  <div class="ecg-wrap">
    <svg viewBox="0 0 1200 80" preserveAspectRatio="none" width="100%" height="80">
      <path class="ecg-path" d="M0,40 L100,40 L120,40 L130,10 L140,70 L150,5 L160,75 L170,40 L280,40 L300,40 L310,10 L320,70 L330,5 L340,75 L350,40 L460,40 L480,40 L490,10 L500,70 L510,5 L520,75 L530,40 L640,40 L660,40 L670,10 L680,70 L690,5 L700,75 L710,40 L820,40 L840,40 L850,10 L860,70 L870,5 L880,75 L890,40 L1000,40 L1020,40 L1030,10 L1040,70 L1050,5 L1060,75 L1070,40 L1200,40"/>
    </svg>
  </div>
</section>

<!-- STATS -->
<div class="stats">
  <div class="stat">
    <div class="stat-val" id="s-patients">—</div>
    <div class="stat-label">Пациенттер</div>
  </div>
  <div class="stat">
    <div class="stat-val red" id="s-endpoints">7</div>
    <div class="stat-label">Эндпоинттар</div>
  </div>
  <div class="stat">
    <div class="stat-val green" id="s-uptime">—</div>
    <div class="stat-label">Uptime (сек)</div>
  </div>
  <div class="stat">
    <div class="stat-val blue">2.2.0</div>
    <div class="stat-label">Версия</div>
  </div>
  <div class="stat">
    <div class="stat-val orange">184</div>
    <div class="stat-label">Тесттер ✓</div>
  </div>
</div>

<!-- ENDPOINTS -->
<section class="section" id="endpoints">
  <div class="section-head">
    <span class="section-label">// endpoints</span>
    <h2 class="section-title">API эндпоинттары</h2>
  </div>
  <div class="grid">

    <div class="card" style="--accent:var(--red)">
      <div class="card-top">
        <span class="method post">POST</span>
        <span class="path">/ml/risk-classification</span>
      </div>
      <p class="card-desc">ACC/AHA риск классификациясы. 5 топ: Норма A → Стадия D. BMI, коморбидность, confidence.</p>
      <div class="card-tags"><span class="tag">PatientData</span><span class="tag">RiskResponse</span><span class="tag">score 0–1</span></div>
    </div>

    <div class="card" style="--accent:var(--orange)">
      <div class="card-top">
        <span class="method post">POST</span>
        <span class="path">/ml/check-alerts</span>
      </div>
      <p class="card-desc">Клиникалық алерттер. ALERT-01..06, COMBO-01/05, TREND-01..04. Приоритет CRITICAL → LOW.</p>
      <div class="card-tags"><span class="tag">ALERT-03 Hb</span><span class="tag">ALERT-04 Cr</span><span class="tag">ALERT-06 ЭКГ</span></div>
    </div>

    <div class="card" style="--accent:var(--blue)">
      <div class="card-top">
        <span class="method post">POST</span>
        <span class="path">/ml/ai-summary</span>
      </div>
      <p class="card-desc">AI клиникалық заключение. Риск + алерттер + динамика + ұсыныс. ИИ диагноз қоймайды.</p>
      <div class="card-tags"><span class="tag">SummaryRequest</span><span class="tag">visit_history</span></div>
    </div>

    <div class="card" style="--accent:var(--yellow)">
      <div class="card-top">
        <span class="method post">POST</span>
        <span class="path">/ml/dynamics-analysis</span>
      </div>
      <p class="card-desc">Тренд анализі. Сызықтық регрессия, болжам, velocity, severity. Минимум 2 визит.</p>
      <div class="card-tags"><span class="tag">linear_regression</span><span class="tag">forecast</span><span class="tag">severity</span></div>
    </div>

    <div class="card" style="--accent:var(--red)">
      <div class="card-top">
        <span class="method post">POST</span>
        <span class="path">/ml/nt-probnp-paradox</span>
      </div>
      <p class="card-desc">NT-proBNP парадоксі. NT &lt;125 + ФВ &lt;35% + симптомдар → ожирение немесе D стадиясы.</p>
      <div class="card-tags"><span class="tag">paradox detection</span><span class="tag">ожирение</span></div>
    </div>

    <div class="card" style="--accent:var(--green)">
      <div class="card-top">
        <span class="method post">POST</span>
        <span class="path">/ml/import/classify</span>
      </div>
      <p class="card-desc">Excel датасет импорты. Batch классификация, алерттер, статистика. .xlsx / .xls.</p>
      <div class="card-tags"><span class="tag">UploadFile</span><span class="tag">134 пациент</span><span class="tag">batch</span></div>
    </div>

    <div class="card" style="--accent:var(--blue)">
      <div class="card-top">
        <span class="method post">POST</span>
        <span class="path">/ml/save-training-data</span>
      </div>
      <p class="card-desc">Training data жинау. Дәрігер тағайындауы → v3.0 XGBoost моделінің негізі. Gold label.</p>
      <div class="card-tags"><span class="tag">gold_label</span><span class="tag">v3.0 ML</span><span class="tag">JSONL</span></div>
    </div>

    <div class="card" style="--accent:var(--green)">
      <div class="card-top">
        <span class="method get">GET</span>
        <span class="path">/health</span>
      </div>
      <p class="card-desc">Spring Boot health check. Статус, версия, uptime секундтары. Мониторинг үшін.</p>
      <div class="card-tags"><span class="tag">Spring Boot</span><span class="tag">uptime</span></div>
    </div>

  </div>
</section>

<!-- RISK BANDS -->
<section class="risk-section" id="risk">
  <div class="risk-inner">
    <div class="section-head">
      <span class="section-label">// acc/aha</span>
      <h2 class="section-title">Риск топтары</h2>
    </div>
    <div class="risk-grid">
      <div class="risk-band a">
        <div class="risk-color"></div>
        <div class="risk-name">норма</div>
        <div class="risk-desc">Симптом жоқ немесе жүрек зақымы бар (симптомсыз). Жылдық бақылау.</div>
      </div>
      <div class="risk-band c">
        <div class="risk-color"></div>
        <div class="risk-name">C</div>
        <div class="risk-desc">Жүрек зақымы + ХСН симптомдары. 3 айда бір бақылау.</div>
      </div>
      <div class="risk-band cd">
        <div class="risk-color"></div>
        <div class="risk-name">C → D</div>
        <div class="risk-desc">Ауыр симптомдар, терапияға төзімді. 2 аптада бір бақылау.</div>
      </div>
      <div class="risk-band d">
        <div class="risk-color"></div>
        <div class="risk-name">D</div>
        <div class="risk-desc">Рефрактерлі ХСН. Жедел госпитализация немесе паллиативтік көмек.</div>
      </div>
    </div>
  </div>
</section>

<!-- TERMINAL -->
<section class="terminal-section">
  <div class="section-head">
    <span class="section-label">// demo</span>
    <h2 class="section-title">Жылдам тест</h2>
  </div>
  <div class="terminal">
    <div class="term-head">
      <div class="term-dot r"></div>
      <div class="term-dot y"></div>
      <div class="term-dot g"></div>
      <span class="term-title">cardiotracker — bash</span>
    </div>
    <div class="term-body">
      <div class="term-line"><span class="t-prompt">❯</span><span class="t-cmd">curl -X POST /ml/risk-classification \</span></div>
      <div class="term-line"><span class="t-prompt"> </span><span class="t-dim">  -d '{"ef": 38.0, "nt_probnp": 1200.0, "six_min_walk": 280.0,</span></div>
      <div class="term-line"><span class="t-prompt"> </span><span class="t-dim">       "has_prior_mi": true, "symptom_class": 3}'</span></div>
      <div class="term-line"><span class="t-prompt"> </span><span class="t-dim">  </span></div>
      <div class="term-line"><span class="t-prompt">→</span><span class="t-yellow">  "risk_group": "C",</span></div>
      <div class="term-line"><span class="t-prompt"> </span><span class="t-yellow">  "risk_score": 0.48,</span></div>
      <div class="term-line"><span class="t-prompt"> </span><span class="t-blue">  "confidence": "высокая",  "risk_color": "yellow"</span></div>
    </div>
  </div>
</section>

<!-- FOOTER -->
<footer>
  <div class="footer-left">
    CardioTracker ML <span>v2.2.0</span> · Zhambyl Hub × Кардиохирургиялық орталық · <span>Тараз, 2026</span>
  </div>
  <div class="footer-right">
    <a href="/docs" class="footer-link">Swagger</a>
    <a href="/health" class="footer-link">Health</a>
    <a href="/ml/training-data/stats" class="footer-link">Stats</a>
  </div>
</footer>

<div class="version">v2.2.0 · ML</div>

<script>
// Health check — uptime жүктеу
fetch('/health').then(r=>r.json()).then(d=>{
  document.getElementById('s-uptime').textContent = d.uptime_seconds ?? '—';
}).catch(()=>{});

// Animate stat numbers
function animateNum(el, target, duration=1200){
  let start=0, step=target/60;
  let timer=setInterval(()=>{
    start+=step;
    if(start>=target){start=target;clearInterval(timer)}
    el.textContent=Math.round(start);
  }, duration/60);
}
setTimeout(()=>{
  animateNum(document.getElementById('s-patients'), 500);
}, 400);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return HTMLResponse(content=HTML)


@app.get("/health", tags=["Система"])
async def health():
    return {
        "status": "ok",
        "service": "CardioTracker ML",
        "version": "2.2.0",
        "uptime_seconds": round(time.time() - _START_TIME),
    }