"""
routers/explorer.py — Custom API Explorer
CardioTracker ML v2.2

Swagger орнына өзіміздің интерактивті API Explorer.
Дәрігер мен медбике үшін ыңғайлы интерфейс.

Endpoints:
  GET /explorer — Explorer UI
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

EXPLORER_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CardioTracker — API Explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0e14;--s1:#0f1419;--s2:#141b24;--s3:#1a2332;
  --border:rgba(255,255,255,.07);--border2:rgba(255,255,255,.12);
  --text:#dce7f3;--muted:#4d6278;--dim:#2a3a4a;
  --red:#e8404a;--green:#2ecc8a;--blue:#4a9eff;
  --yellow:#f0b429;--orange:#f07842;--purple:#9b7aff;
  --mono:'JetBrains Mono',monospace;
  --sans:'Inter',sans-serif;
}
html,body{height:100%;overflow:hidden}
body{font-family:var(--sans);background:var(--bg);color:var(--text);display:flex;flex-direction:column}

/* ── TOPBAR ── */
.topbar{
  height:52px;flex-shrink:0;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;
  background:var(--s1);
  border-bottom:1px solid var(--border);
  z-index:10;
}
.tb-left{display:flex;align-items:center;gap:16px}
.tb-logo{display:flex;align-items:center;gap:9px}
.tb-icon{
  width:28px;height:28px;border-radius:7px;
  background:var(--red);
  display:grid;place-items:center;font-size:14px;
  position:relative;
}
.tb-icon::after{
  content:'';position:absolute;inset:0;border-radius:7px;
  box-shadow:0 0 16px rgba(232,64,74,.5);
}
.tb-name{font-size:13px;font-weight:600;letter-spacing:-.01em}
.tb-name span{color:var(--muted);font-weight:400}
.tb-sep{width:1px;height:20px;background:var(--border)}
.tb-ver{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.05em}
.tb-right{display:flex;align-items:center;gap:12px}
.tb-status{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:10px;color:var(--green)}
.tb-status::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.tb-btn{
  font-family:var(--mono);font-size:10px;font-weight:700;
  padding:5px 12px;border-radius:5px;
  text-decoration:none;letter-spacing:.05em;
  transition:all .15s;
}
.tb-btn.outline{color:var(--muted);border:1px solid var(--border)}
.tb-btn.outline:hover{color:var(--text);border-color:var(--border2)}
.tb-btn.primary{background:var(--red);color:#fff}
.tb-btn.primary:hover{background:#f05560}

/* ── LAYOUT ── */
.layout{display:flex;flex:1;overflow:hidden}

/* ── SIDEBAR ── */
.sidebar{
  width:240px;flex-shrink:0;
  background:var(--s1);
  border-right:1px solid var(--border);
  overflow-y:auto;
  display:flex;flex-direction:column;
}
.sb-section{padding:16px 12px 8px;font-family:var(--mono);font-size:9px;letter-spacing:.12em;color:var(--muted);text-transform:uppercase}
.ep-item{
  display:flex;align-items:center;gap:10px;
  padding:9px 12px;cursor:pointer;
  border-left:2px solid transparent;
  transition:all .15s;
  position:relative;
}
.ep-item:hover{background:var(--s2)}
.ep-item.active{background:var(--s2);border-left-color:var(--red)}
.ep-badge{
  font-family:var(--mono);font-size:8px;font-weight:700;
  padding:2px 5px;border-radius:3px;flex-shrink:0;
  letter-spacing:.03em;
}
.post{background:rgba(232,64,74,.15);color:var(--red)}
.get{background:rgba(46,204,138,.15);color:var(--green)}
.ep-name{font-size:12px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ep-name.active{color:var(--text)}
.ep-item:not(.active) .ep-name{color:var(--muted)}
.sb-divider{height:1px;background:var(--border);margin:8px 12px}

/* ── MAIN PANEL ── */
.main{flex:1;display:flex;overflow:hidden}

/* ── EDITOR PANE ── */
.editor-pane{
  width:420px;flex-shrink:0;
  background:var(--s1);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;
  overflow:hidden;
}
.pane-head{
  padding:16px 20px 12px;
  border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.pane-title{font-size:14px;font-weight:600;margin-bottom:4px}
.pane-path{font-family:var(--mono);font-size:11px;color:var(--muted)}
.pane-desc{font-size:12px;color:var(--muted);margin-top:8px;line-height:1.6}
.pane-body{flex:1;overflow-y:auto;padding:16px 20px}

/* ── FORM ── */
.field-group{margin-bottom:20px}
.field-group-title{
  font-family:var(--mono);font-size:9px;font-weight:700;
  letter-spacing:.12em;color:var(--muted);text-transform:uppercase;
  margin-bottom:10px;padding-bottom:6px;
  border-bottom:1px solid var(--border);
}
.field{margin-bottom:12px}
.field label{
  display:flex;align-items:center;justify-content:space-between;
  font-size:11px;font-weight:500;margin-bottom:5px;
}
.field label span{color:var(--muted);font-family:var(--mono);font-size:10px}
.field input,.field select{
  width:100%;height:34px;
  background:var(--s2);
  border:1px solid var(--border);
  border-radius:6px;
  color:var(--text);
  font-family:var(--mono);font-size:12px;
  padding:0 10px;
  transition:border-color .15s;
  outline:none;
}
.field input:focus,.field select:focus{border-color:var(--blue)}
.field input[type=checkbox]{width:auto;height:auto;accent-color:var(--red)}
.check-row{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:12px;color:var(--muted)}
.check-row input{flex-shrink:0}
.field-file{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:8px;
  border:1.5px dashed var(--border2);
  border-radius:8px;padding:28px 20px;
  cursor:pointer;transition:all .2s;
  text-align:center;
}
.field-file:hover{border-color:var(--blue);background:rgba(74,158,255,.04)}
.field-file.has-file{border-color:var(--green);background:rgba(46,204,138,.04)}
.field-file-icon{font-size:24px;opacity:.4}
.field-file-txt{font-size:12px;color:var(--muted)}
.field-file-name{font-family:var(--mono);font-size:11px;color:var(--green)}
.pane-footer{padding:16px 20px;border-top:1px solid var(--border);flex-shrink:0}
.run-btn{
  width:100%;height:38px;border-radius:7px;
  background:var(--red);color:#fff;
  font-family:var(--mono);font-size:12px;font-weight:700;
  letter-spacing:.05em;text-transform:uppercase;
  cursor:pointer;border:none;
  transition:all .2s;display:flex;align-items:center;justify-content:center;gap:8px;
}
.run-btn:hover{background:#f05560;transform:translateY(-1px)}
.run-btn:active{transform:translateY(0)}
.run-btn.loading{background:var(--dim);cursor:not-allowed;transform:none}
.run-btn svg{animation:none}
.run-btn.loading svg{animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── RESULT PANE ── */
.result-pane{
  flex:1;
  background:var(--bg);
  display:flex;flex-direction:column;
  overflow:hidden;
}
.result-head{
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 20px;
  border-bottom:1px solid var(--border);
  background:var(--s1);
  flex-shrink:0;
}
.result-meta{display:flex;align-items:center;gap:12px}
.result-title{font-size:12px;font-weight:500;color:var(--muted)}
.status-badge{
  font-family:var(--mono);font-size:10px;font-weight:700;
  padding:3px 8px;border-radius:4px;
}
.s200{background:rgba(46,204,138,.15);color:var(--green)}
.s400{background:rgba(240,180,41,.15);color:var(--yellow)}
.s500{background:rgba(232,64,74,.15);color:var(--red)}
.duration-badge{font-family:var(--mono);font-size:10px;color:var(--muted)}
.copy-btn{
  font-family:var(--mono);font-size:10px;
  padding:4px 10px;border-radius:5px;
  background:transparent;color:var(--muted);
  border:1px solid var(--border);cursor:pointer;
  transition:all .15s;
}
.copy-btn:hover{color:var(--text);border-color:var(--border2)}
.result-body{flex:1;overflow-y:auto;padding:20px}

/* ── RESULT CARDS ── */
.risk-card{
  border-radius:10px;padding:20px;margin-bottom:16px;
  border:1px solid var(--border);
  background:var(--s1);
}
.risk-card-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.risk-label{font-size:20px;font-weight:600}
.risk-score-wrap{text-align:right}
.risk-score{font-family:var(--mono);font-size:28px;font-weight:700}
.risk-score-label{font-size:10px;color:var(--muted)}
.risk-bar-wrap{height:6px;background:var(--s3);border-radius:3px;margin-bottom:16px;overflow:hidden}
.risk-bar{height:100%;border-radius:3px;transition:width .6s ease}
.risk-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.risk-stat{background:var(--s2);border-radius:7px;padding:10px 12px}
.risk-stat-label{font-size:10px;color:var(--muted);margin-bottom:4px}
.risk-stat-val{font-family:var(--mono);font-size:13px;font-weight:500}
.factors-list{margin-top:12px}
.factor-item{
  display:flex;align-items:center;gap:8px;
  padding:6px 0;border-bottom:1px solid var(--border);
  font-size:12px;color:var(--muted);
}
.factor-item:last-child{border-bottom:none}
.factor-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}

.alert-card{
  border-radius:8px;padding:14px 16px;
  margin-bottom:8px;
  border-left:3px solid;
  background:var(--s1);
}
.alert-card.CRITICAL{border-color:var(--red);background:rgba(232,64,74,.06)}
.alert-card.HIGH{border-color:var(--orange);background:rgba(240,120,66,.06)}
.alert-card.MEDIUM{border-color:var(--yellow);background:rgba(240,180,41,.06)}
.alert-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.alert-code{font-family:var(--mono);font-size:10px;font-weight:700}
.alert-priority{font-family:var(--mono);font-size:9px;padding:2px 6px;border-radius:3px}
.pri-CRITICAL{background:rgba(232,64,74,.2);color:var(--red)}
.pri-HIGH{background:rgba(240,120,66,.2);color:var(--orange)}
.pri-MEDIUM{background:rgba(240,180,41,.2);color:var(--yellow)}
.alert-msg{font-size:12px;color:var(--muted);line-height:1.5}

.import-summary{
  background:var(--s1);border:1px solid var(--border);
  border-radius:10px;padding:20px;margin-bottom:16px;
}
.import-dist{display:flex;gap:1px;border-radius:6px;overflow:hidden;height:32px;margin:12px 0}
.dist-seg{display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:10px;font-weight:700;color:#fff;transition:flex .3s}
.dist-a{background:#2ecc8a}
.dist-b{background:#4a9eff}
.dist-c{background:#f0b429}
.dist-cd{background:#f07842}
.dist-d{background:#e8404a}
.import-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.imp-stat{background:var(--s2);border-radius:7px;padding:10px;text-align:center}
.imp-stat-val{font-family:var(--mono);font-size:18px;font-weight:700;margin-bottom:2px}
.imp-stat-label{font-size:10px;color:var(--muted)}
.patient-table{width:100%;border-collapse:collapse;font-size:11px}
.patient-table th{
  text-align:left;padding:8px 10px;
  font-family:var(--mono);font-size:9px;letter-spacing:.1em;color:var(--muted);
  border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--s1);
}
.patient-table td{padding:7px 10px;border-bottom:1px solid var(--border);color:var(--muted)}
.patient-table td:first-child{color:var(--text);font-weight:500}
.patient-table tr:hover td{background:var(--s2)}
.rg-pill{
  display:inline-flex;align-items:center;gap:4px;
  font-family:var(--mono);font-size:9px;font-weight:700;
  padding:2px 7px;border-radius:4px;
}
.rg-green{background:rgba(46,204,138,.12);color:var(--green)}
.rg-blue{background:rgba(74,158,255,.12);color:var(--blue)}
.rg-yellow{background:rgba(240,180,41,.12);color:var(--yellow)}
.rg-orange{background:rgba(240,120,66,.12);color:var(--orange)}
.rg-red{background:rgba(232,64,74,.12);color:var(--red)}

.json-pre{
  background:var(--s1);border:1px solid var(--border);
  border-radius:8px;padding:16px;
  font-family:var(--mono);font-size:12px;line-height:1.7;
  overflow-x:auto;white-space:pre;
}
.j-key{color:#9b7aff}
.j-str{color:#2ecc8a}
.j-num{color:#f0b429}
.j-bool{color:#4a9eff}
.j-null{color:var(--muted)}

.empty-state{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:100%;gap:12px;color:var(--muted);
}
.empty-icon{font-size:40px;opacity:.2}
.empty-text{font-size:13px}
.empty-sub{font-size:11px;font-family:var(--mono)}

/* ── SUMMARY TEXT ── */
.summary-block{
  background:var(--s1);border:1px solid var(--border);
  border-radius:10px;padding:20px;font-size:13px;
  line-height:1.8;color:var(--muted);white-space:pre-wrap;
}
.summary-block .section-header{color:var(--text);font-weight:600;font-family:var(--mono);font-size:11px;letter-spacing:.05em}

scrollbar-width:thin;scrollbar-color:var(--dim) transparent;
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--dim);border-radius:2px}
</style>
</head>
<body>

<!-- TOPBAR -->
<div class="topbar">
  <div class="tb-left">
    <div class="tb-logo">
      <div class="tb-icon">🫀</div>
      <span class="tb-name">CardioTracker <span>/ API Explorer</span></span>
    </div>
    <div class="tb-sep"></div>
    <span class="tb-ver">v2.2.0</span>
  </div>
  <div class="tb-right">
    <div class="tb-status">Online</div>
    <a href="/" class="tb-btn outline">← Басты бет</a>
    <a href="/docs" class="tb-btn outline">Swagger</a>
  </div>
</div>

<div class="layout">

  <!-- SIDEBAR -->
  <aside class="sidebar">
    <div class="sb-section">Классификация</div>
    <div class="ep-item active" onclick="selectEp('risk')" id="ep-risk">
      <span class="ep-badge post">POST</span>
      <span class="ep-name">risk-classification</span>
    </div>
    <div class="ep-item" onclick="selectEp('alerts')" id="ep-alerts">
      <span class="ep-badge post">POST</span>
      <span class="ep-name">check-alerts</span>
    </div>
    <div class="sb-divider"></div>
    <div class="sb-section">AI сервистер</div>
    <div class="ep-item" onclick="selectEp('summary')" id="ep-summary">
      <span class="ep-badge post">POST</span>
      <span class="ep-name">ai-summary</span>
    </div>
    <div class="sb-divider"></div>
    <div class="sb-section">Импорт</div>
    <div class="ep-item" onclick="selectEp('import')" id="ep-import">
      <span class="ep-badge post">POST</span>
      <span class="ep-name">import/classify</span>
    </div>
    <div class="sb-divider"></div>
    <div class="sb-section">Жүйе</div>
    <div class="ep-item" onclick="selectEp('health')" id="ep-health">
      <span class="ep-badge get">GET</span>
      <span class="ep-name">health</span>
    </div>
    <div class="ep-item" onclick="selectEp('stats')" id="ep-stats">
      <span class="ep-badge get">GET</span>
      <span class="ep-name">training-data/stats</span>
    </div>
  </aside>

  <!-- MAIN -->
  <div class="main">

    <!-- EDITOR PANE -->
    <div class="editor-pane">
      <div class="pane-head">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
          <span class="ep-badge post" id="form-method">POST</span>
          <span class="pane-path" id="form-path">/ml/risk-classification</span>
        </div>
        <div class="pane-title" id="form-title">Риск классификациясы</div>
        <div class="pane-desc" id="form-desc">Пациент деректерін енгізіп, ACC/AHA риск тобын анықтаңыз.</div>
      </div>
      <div class="pane-body" id="form-body"><!-- JS renders form --></div>
      <div class="pane-footer">
        <button class="run-btn" id="run-btn" onclick="runRequest()">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          <span id="run-label">Жіберу</span>
        </button>
      </div>
    </div>

    <!-- RESULT PANE -->
    <div class="result-pane">
      <div class="result-head">
        <div class="result-meta">
          <span class="result-title">Жауап</span>
          <span class="status-badge" id="status-badge" style="display:none"></span>
          <span class="duration-badge" id="duration-badge"></span>
        </div>
        <button class="copy-btn" onclick="copyResult()" id="copy-btn" style="display:none">Көшіру</button>
      </div>
      <div class="result-body" id="result-body">
        <div class="empty-state">
          <div class="empty-icon">◎</div>
          <div class="empty-text">Сұраныс жіберілмеген</div>
          <div class="empty-sub">Сол жақтан endpoint таңдап, «Жіберу» батырмасын басыңыз</div>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
const BASE = window.location.origin;
let currentEp = 'risk';
let lastResult = null;
let uploadedFile = null;

const ENDPOINTS = {
  risk: {
    method:'POST', path:'/ml/risk-classification',
    title:'Риск классификациясы',
    desc:'Пациент деректерін енгізіп, ACC/AHA риск тобын анықтаңыз.',
    form: 'riskForm'
  },
  alerts: {
    method:'POST', path:'/ml/check-alerts',
    title:'Клиникалық алерттер',
    desc:'Визит деректері бойынша алерттерді тексеріңіз.',
    form: 'alertsForm'
  },
  summary: {
    method:'POST', path:'/ml/ai-summary',
    title:'AI Клиникалық заключение',
    desc:'Пациент деректері бойынша автоматты сводка жасаңыз.',
    form: 'summaryForm'
  },
  import: {
    method:'POST', path:'/ml/import/classify',
    title:'Excel датасет импорты',
    desc:'Excel файлын жүктеп, барлық пациенттерді классификациялаңыз.',
    form: 'importForm'
  },
  health: {
    method:'GET', path:'/health',
    title:'Health Check',
    desc:'Сервистің жұмыс күйін тексеріңіз.',
    form: 'healthForm'
  },
  stats: {
    method:'GET', path:'/ml/training-data/stats',
    title:'Training Data статистикасы',
    desc:'Жиналған оқыту деректерінің санын және үлестірімін қараңыз.',
    form: 'statsForm'
  }
};

const FORMS = {
  riskForm: `
<div class="field-group">
  <div class="field-group-title">Кардиология</div>
  <div class="field"><label>ФВ/ЛЖ, % <span>ef</span></label><input type="number" id="f_ef" placeholder="38.0" step="0.1"></div>
  <div class="field"><label>NT-proBNP, пг/мл <span>nt_probnp</span></label><input type="number" id="f_nt" placeholder="1200"></div>
  <div class="field"><label>Тест 6 мин, м <span>six_min_walk</span></label><input type="number" id="f_walk" placeholder="280"></div>
  <div class="field"><label>NYHA ФК <span>symptom_class</span></label>
    <select id="f_nyha"><option value="">—</option><option value="1">ФК I</option><option value="2">ФК II</option><option value="3">ФК III</option><option value="4">ФК IV</option></select>
  </div>
</div>
<div class="field-group">
  <div class="field-group-title">Биохимия</div>
  <div class="field"><label>Гемоглобин, г/л <span>hemoglobin</span></label><input type="number" id="f_hb" placeholder="108"></div>
  <div class="field"><label>Креатинин, мкмоль/л <span>creatinine</span></label><input type="number" id="f_cr" placeholder="95"></div>
  <div class="field"><label>Мочевина, ммоль/л <span>urea</span></label><input type="number" id="f_urea" placeholder="6.5"></div>
</div>
<div class="field-group">
  <div class="field-group-title">ЭКГ</div>
  <label class="check-row"><input type="checkbox" id="f_af"> Жыбырлау аритмиясы (ФП)</label>
  <label class="check-row"><input type="checkbox" id="f_st"> ST өзгерістері</label>
  <label class="check-row"><input type="checkbox" id="f_block"> Блокада</label>
  <label class="check-row"><input type="checkbox" id="f_tachy"> Тахикардия</label>
</div>
<div class="field-group">
  <div class="field-group-title">Коморбидность</div>
  <label class="check-row"><input type="checkbox" id="f_ht"> Артериалдық гипертония</label>
  <label class="check-row"><input type="checkbox" id="f_mi"> ИМ анамнезінде</label>
  <label class="check-row"><input type="checkbox" id="f_dm"> Қант диабеті</label>
  <label class="check-row"><input type="checkbox" id="f_ckd"> ХБП</label>
</div>
<div class="field-group">
  <div class="field-group-title">Антропометрия</div>
  <div class="field"><label>Бой, см <span>height_cm</span></label><input type="number" id="f_ht2" placeholder="168"></div>
  <div class="field"><label>Салмақ, кг <span>weight_kg</span></label><input type="number" id="f_wt" placeholder="78"></div>
  <div class="field"><label>Жасы <span>age</span></label><input type="number" id="f_age" placeholder="65"></div>
</div>`,

  alertsForm: `
<div class="field-group">
  <div class="field-group-title">Ағымдағы визит</div>
  <div class="field"><label>ФВ/ЛЖ, %</label><input type="number" id="a_ef" placeholder="28.0" step="0.1"></div>
  <div class="field"><label>NT-proBNP, пг/мл</label><input type="number" id="a_nt" placeholder="6000"></div>
  <div class="field"><label>Гемоглобин, г/л</label><input type="number" id="a_hb" placeholder="85"></div>
  <div class="field"><label>Креатинин, мкмоль/л</label><input type="number" id="a_cr" placeholder="180"></div>
  <label class="check-row" style="margin-bottom:6px"><input type="checkbox" id="a_af"> ФП</label>
  <label class="check-row"><input type="checkbox" id="a_st"> ST өзгерістері</label>
</div>
<div class="field-group">
  <div class="field-group-title">Алдыңғы визит (тарих)</div>
  <div class="field"><label>Алдыңғы Креатинин</label><input type="number" id="a_cr_prev" placeholder="110"></div>
  <div class="field"><label>Алдыңғы NT-proBNP</label><input type="number" id="a_nt_prev" placeholder=""></div>
</div>`,

  summaryForm: `
<div class="field-group">
  <div class="field-group-title">Пациент деректері</div>
  <div class="field"><label>ФВ/ЛЖ, %</label><input type="number" id="s_ef" placeholder="38.0" step="0.1"></div>
  <div class="field"><label>NT-proBNP, пг/мл</label><input type="number" id="s_nt" placeholder="1200"></div>
  <div class="field"><label>Тест 6 мин, м</label><input type="number" id="s_walk" placeholder="280"></div>
  <div class="field"><label>Гемоглобин, г/л</label><input type="number" id="s_hb" placeholder="108"></div>
  <div class="field"><label>Креатинин, мкмоль/л</label><input type="number" id="s_cr" placeholder="95"></div>
  <label class="check-row"><input type="checkbox" id="s_af"> ФП</label>
</div>`,

  importForm: `
<div class="field-group">
  <div class="field-group-title">Excel файл</div>
  <div class="field-file" id="drop-zone" onclick="document.getElementById('file-input').click()">
    <div class="field-file-icon">📊</div>
    <div class="field-file-txt">Файлды осында сүйреңіз немесе басыңыз</div>
    <div class="field-file-name" id="file-name" style="display:none"></div>
    <div style="font-size:10px;color:var(--muted);font-family:var(--mono)">.xlsx / .xls</div>
  </div>
  <input type="file" id="file-input" accept=".xlsx,.xls" style="display:none" onchange="handleFile(this)">
</div>`,

  healthForm: `<div style="padding:20px 0;text-align:center;color:var(--muted);font-size:13px">Параметр қажет емес — тікелей жіберіңіз.</div>`,
  statsForm: `<div style="padding:20px 0;text-align:center;color:var(--muted);font-size:13px">Параметр қажет емес — тікелей жіберіңіз.</div>`,
};

function selectEp(ep) {
  currentEp = ep;
  document.querySelectorAll('.ep-item').forEach(el => el.classList.remove('active'));
  const el = document.getElementById('ep-' + ep);
  if (el) el.classList.add('active');

  const cfg = ENDPOINTS[ep];
  const methodEl = document.getElementById('form-method');
  methodEl.textContent = cfg.method;
  methodEl.className = 'ep-badge ' + cfg.method.toLowerCase();
  document.getElementById('form-path').textContent = cfg.path;
  document.getElementById('form-title').textContent = cfg.title;
  document.getElementById('form-desc').textContent = cfg.desc;
  document.getElementById('form-body').innerHTML = FORMS[cfg.form] || '';

  document.getElementById('result-body').innerHTML = `<div class="empty-state"><div class="empty-icon">◎</div><div class="empty-text">Сұраныс жіберілмеген</div></div>`;
  document.getElementById('status-badge').style.display = 'none';
  document.getElementById('duration-badge').textContent = '';
  document.getElementById('copy-btn').style.display = 'none';
  lastResult = null;

  if (ep === 'import') setupDrop();
}

function setupDrop() {
  const zone = document.getElementById('drop-zone');
  if (!zone) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = 'var(--blue)'; });
  zone.addEventListener('dragleave', () => { zone.style.borderColor = ''; });
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });
}

function handleFile(input) {
  if (input.files[0]) setFile(input.files[0]);
}

function setFile(file) {
  uploadedFile = file;
  const zone = document.getElementById('drop-zone');
  const nameEl = document.getElementById('file-name');
  if (zone) zone.classList.add('has-file');
  if (nameEl) { nameEl.textContent = file.name; nameEl.style.display = 'block'; }
  const txt = zone?.querySelector('.field-file-txt');
  if (txt) txt.textContent = 'Файл таңдалды';
}

function val(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  if (el.type === 'checkbox') return el.checked;
  if (el.type === 'number') return el.value ? parseFloat(el.value) : null;
  return el.value || null;
}

function buildBody() {
  const ep = currentEp;
  if (ep === 'risk' || ep === 'summary') {
    const prefix = ep === 'risk' ? 'f_' : 's_';
    const data = {};
    const map = ep === 'risk'
      ? { ef:'f_ef', nt_probnp:'f_nt', six_min_walk:'f_walk', hemoglobin:'f_hb', creatinine:'f_cr', urea:'f_urea',
          ecg_af:'f_af', ecg_st_changes:'f_st', ecg_blockade:'f_block', ecg_tachycardia:'f_tachy',
          has_hypertension:'f_ht', has_prior_mi:'f_mi', has_diabetes:'f_dm', has_ckd:'f_ckd',
          height_cm:'f_ht2', weight_kg:'f_wt', age:'f_age',
          symptom_class: 'f_nyha'
        }
      : { ef:'s_ef', nt_probnp:'s_nt', six_min_walk:'s_walk', hemoglobin:'s_hb', creatinine:'s_cr', ecg_af:'s_af' };

    for (const [key, id] of Object.entries(map)) {
      const v = val(id);
      if (v !== null && v !== '') {
        if (key === 'symptom_class') data[key] = v ? parseInt(v) : null;
        else data[key] = v;
      }
    }
    if (ep === 'summary') return { patient_data: data, visit_history: [] };
    return data;
  }
  if (ep === 'alerts') {
    const current = {};
    const flds = { ef:'a_ef', nt_probnp:'a_nt', hemoglobin:'a_hb', creatinine:'a_cr', ecg_af:'a_af', ecg_st_changes:'a_st' };
    for (const [k,id] of Object.entries(flds)) { const v = val(id); if (v !== null) current[k] = v; }
    const hist = [];
    const crPrev = val('a_cr_prev'); const ntPrev = val('a_nt_prev');
    if (crPrev || ntPrev) {
      const h = {};
      if (crPrev) h.creatinine = crPrev;
      if (ntPrev) h.nt_probnp = ntPrev;
      hist.push(h);
    }
    return { current, visit_history: hist };
  }
  return null;
}

async function runRequest() {
  const cfg = ENDPOINTS[currentEp];
  const btn = document.getElementById('run-btn');
  const label = document.getElementById('run-label');
  btn.classList.add('loading');
  label.textContent = 'Жүктелуде...';

  const t0 = Date.now();
  try {
    let resp;
    if (cfg.method === 'GET') {
      resp = await fetch(BASE + cfg.path);
    } else if (currentEp === 'import') {
      if (!uploadedFile) { alert('Файл таңдаңыз'); btn.classList.remove('loading'); label.textContent='Жіберу'; return; }
      const fd = new FormData();
      fd.append('file', uploadedFile);
      resp = await fetch(BASE + cfg.path, { method:'POST', body: fd });
    } else {
      const body = buildBody();
      resp = await fetch(BASE + cfg.path, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    }
    const dur = Date.now() - t0;
    const data = await resp.json();
    lastResult = data;

    const sb = document.getElementById('status-badge');
    sb.textContent = resp.status;
    sb.className = 'status-badge s' + (resp.status < 300 ? '200' : resp.status < 500 ? '400' : '500');
    sb.style.display = '';
    document.getElementById('duration-badge').textContent = dur + 'ms';
    document.getElementById('copy-btn').style.display = '';

    renderResult(currentEp, data, resp.status);
  } catch(e) {
    document.getElementById('result-body').innerHTML = `<div class="json-pre" style="color:var(--red)">${e.message}</div>`;
  }
  btn.classList.remove('loading');
  label.textContent = 'Жіберу';
}

function colorMap(rg) {
  const m = {'Норма (A)':'rg-green','Стадия B':'rg-blue','Риск C':'rg-yellow','C→D':'rg-orange','Стадия D':'rg-red'};
  return m[rg] || 'rg-blue';
}
function scoreColor(score) {
  if (score < 0.15) return 'var(--green)';
  if (score < 0.35) return 'var(--blue)';
  if (score < 0.55) return 'var(--yellow)';
  if (score < 0.75) return 'var(--orange)';
  return 'var(--red)';
}

function renderResult(ep, data, status) {
  const body = document.getElementById('result-body');
  if (status >= 400) {
    body.innerHTML = `<div class="json-pre" style="color:var(--red)">${JSON.stringify(data,null,2)}</div>`;
    return;
  }

  if (ep === 'risk') {
    const sc = data.risk_score || 0;
    const clr = scoreColor(sc);
    const factors = (data.contributing_factors || []).slice(0,5);
    const warnings = (data.warnings || []).slice(0,3);
    body.innerHTML = `
<div class="risk-card">
  <div class="risk-card-top">
    <div>
      <div style="font-size:10px;color:var(--muted);margin-bottom:4px;font-family:var(--mono)">РИСК ТОБЫ</div>
      <div class="risk-label" style="color:${clr}">${data.risk_group}</div>
    </div>
    <div class="risk-score-wrap">
      <div class="risk-score" style="color:${clr}">${sc.toFixed(2)}</div>
      <div class="risk-score-label">/ 1.00 балл</div>
    </div>
  </div>
  <div class="risk-bar-wrap"><div class="risk-bar" style="width:${sc*100}%;background:${clr}"></div></div>
  <div class="risk-grid">
    <div class="risk-stat"><div class="risk-stat-label">Сенімділік</div><div class="risk-stat-val" style="color:var(--blue)">${data.confidence||'—'}</div></div>
    <div class="risk-stat"><div class="risk-stat-label">Толықтық</div><div class="risk-stat-val">${data.confidence_pct||0}%</div></div>
    <div class="risk-stat"><div class="risk-stat-label">BMI</div><div class="risk-stat-val">${data.bmi||'—'}</div></div>
    <div class="risk-stat"><div class="risk-stat-label">Ұсыныс</div><div class="risk-stat-val" style="font-size:10px;line-height:1.4">${(data.recommendation||'').slice(0,40)}...</div></div>
  </div>
  ${factors.length ? `<div class="factors-list">${factors.map(f=>`<div class="factor-item"><div class="factor-dot" style="background:${clr}"></div>${f}</div>`).join('')}</div>` : ''}
  ${warnings.length ? `<div style="margin-top:12px">${warnings.map(w=>`<div style="font-size:11px;color:var(--orange);padding:4px 0;border-bottom:1px solid var(--border)">⚠ ${w}</div>`).join('')}</div>` : ''}
</div>
${renderJSON(data)}`;
  }

  else if (ep === 'alerts') {
    const alerts = data.alerts || [];
    const summary = `
<div class="import-stats" style="margin-bottom:16px">
  <div class="imp-stat"><div class="imp-stat-val" style="color:var(--red)">${data.total_alerts||0}</div><div class="imp-stat-label">Барлық алерт</div></div>
  <div class="imp-stat"><div class="imp-stat-val" style="color:var(--red)">${data.critical_count||0}</div><div class="imp-stat-label">CRITICAL</div></div>
  <div class="imp-stat"><div class="imp-stat-val" style="color:var(--orange)">${data.high_count||0}</div><div class="imp-stat-label">HIGH</div></div>
</div>`;
    if (!alerts.length) {
      body.innerHTML = summary + `<div class="empty-state" style="height:200px"><div class="empty-icon">✓</div><div class="empty-text" style="color:var(--green)">Алерт жоқ</div></div>`;
    } else {
      body.innerHTML = summary + alerts.map(a=>`
<div class="alert-card ${a.priority}">
  <div class="alert-top">
    <span class="alert-code" style="color:var(--text)">${a.alert_code}</span>
    <span class="alert-priority pri-${a.priority}">${a.priority}</span>
  </div>
  <div class="alert-msg">${a.message}</div>
  ${a.value!=null ? `<div style="margin-top:6px;font-family:var(--mono);font-size:10px;color:var(--muted)">Мән: ${a.value} / Шекара: ${a.threshold}</div>` : ''}
</div>`).join('');
    }
  }

  else if (ep === 'import') {
    const dist = data.distribution || {};
    const total = data.total_patients || 1;
    const segs = [
      ['Норма (A)','dist-a'], ['Стадия B','dist-b'],
      ['Риск C','dist-c'], ['C→D','dist-cd'], ['Стадия D','dist-d']
    ].map(([k,cls]) => {
      const cnt = dist[k]||0;
      const pct = cnt/total*100;
      return pct > 2 ? `<div class="dist-seg ${cls}" style="flex:${pct}">${cnt}</div>` : '';
    }).join('');

    const rows = (data.results||[]).slice(0,50).map(r=>`
<tr>
  <td>${r.patient_name||'—'}</td>
  <td><span class="rg-pill ${colorMap(r.risk_group)}">${r.risk_group}</span></td>
  <td>${(r.risk_score||0).toFixed(2)}</td>
  <td>${r.nyha_class ? 'ФК '+r.nyha_class : '—'}</td>
  <td>${r.alert_count||0}</td>
  <td style="font-family:var(--mono);font-size:9px">${r.ef_lv||'—'}</td>
</tr>`).join('');

    body.innerHTML = `
<div class="import-summary">
  <div style="font-size:12px;color:var(--muted);margin-bottom:8px">Риск тобы бойынша үлестірім</div>
  <div class="import-dist">${segs}</div>
  <div class="import-stats">
    <div class="imp-stat"><div class="imp-stat-val">${total}</div><div class="imp-stat-label">Пациент</div></div>
    <div class="imp-stat"><div class="imp-stat-val" style="color:var(--yellow)">${data.patients_with_alerts||0}</div><div class="imp-stat-label">Алерттері бар</div></div>
    <div class="imp-stat"><div class="imp-stat-val" style="color:var(--red)">${data.patients_critical||0}</div><div class="imp-stat-label">CRITICAL</div></div>
  </div>
</div>
<div style="overflow-x:auto;border:1px solid var(--border);border-radius:8px;background:var(--s1)">
  <table class="patient-table">
    <thead><tr><th>ФИО</th><th>Риск тобы</th><th>Балл</th><th>NYHA</th><th>Алерт</th><th>ФВ%</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
</div>`;
  }

  else if (ep === 'summary') {
    const txt = (data.summary_text||'').replace(
      /══[^═]+══+/g,
      m => `<span class="section-header">${m}</span>`
    );
    body.innerHTML = `
<div style="margin-bottom:12px;display:flex;gap:8px;align-items:center">
  <span class="rg-pill ${colorMap(data.risk_group)}">${data.risk_group||'—'}</span>
  <span style="font-size:11px;color:var(--muted)">Алерттер: ${data.alert_count||0}</span>
</div>
<div class="summary-block">${txt}</div>`;
  }

  else {
    body.innerHTML = renderJSON(data);
  }
}

function renderJSON(data) {
  const str = JSON.stringify(data, null, 2)
    .replace(/("[\w_]+")\s*:/g, '<span class="j-key">$1</span>:')
    .replace(/:\s*(".*?")/g, ': <span class="j-str">$1</span>')
    .replace(/:\s*(\d+\.?\d*)/g, ': <span class="j-num">$1</span>')
    .replace(/:\s*(true|false)/g, ': <span class="j-bool">$1</span>')
    .replace(/:\s*(null)/g, ': <span class="j-null">$1</span>');
  return `<div style="margin-top:12px"><details><summary style="cursor:pointer;font-family:var(--mono);font-size:10px;color:var(--muted);padding:8px 0">Raw JSON ↓</summary><div class="json-pre" style="margin-top:8px">${str}</div></details></div>`;
}

function copyResult() {
  if (lastResult) navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2));
}

selectEp('risk');
</script>
</body>
</html>"""


@router.get("/explorer", response_class=HTMLResponse, include_in_schema=False)
async def explorer():
    return HTMLResponse(content=EXPLORER_HTML)