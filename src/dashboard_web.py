"""Lightweight FastAPI dashboard: two JSON endpoints + one static auto-refreshing page.

No build step, no JS framework — a single HTML file polls /api/state every
`refresh_sec` seconds and re-renders the monitoring tab; the Settings tab
loads /api/strategies once (and after every apply) instead of polling, so
it never fights an in-progress edit. Good enough for a paper-trading bot;
swap in something heavier if you outgrow it.

Single-password auth: set DASHBOARD_PASSWORD in .env to require a password
before anything else loads. Login sets an HMAC-signed, httponly cookie
(derived from the password — no separate secret to configure) that persists
for 30 days, so you don't have to log in again every visit. Leave
DASHBOARD_PASSWORD unset and the dashboard behaves exactly as before (no
auth at all) — this is meant as a quick deterrent for a dashboard exposed
on a server, not a hardened login system; put it behind HTTPS (reverse
proxy) if you want the cookie itself protected in transit too.
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import os
import time
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from .config import AppConfig
from .engine import Engine
from .storage import TRADE_FIELDS

AUTH_COOKIE_NAME = "okx_bot_session"
AUTH_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days — "куки хранились"


def _session_token(password: str) -> str:
    """Deterministic token derived from the password itself (HMAC, not a
    reversible hash of it) — login and verification both just recompute
    this, so there's no separate session store or secret to configure."""
    return hmac.new(password.encode("utf-8"), b"okx-event-bot-authenticated", hashlib.sha256).hexdigest()


class AuthMiddleware(BaseHTTPMiddleware):
    """Gate every request behind a single shared password. No-op entirely
    if DASHBOARD_PASSWORD isn't set (existing no-auth behavior preserved)."""

    def __init__(self, app, password: str):
        super().__init__(app)
        self.password = password
        self.expected_token = _session_token(password) if password else None

    async def dispatch(self, request: Request, call_next):
        if not self.password:
            return await call_next(request)
        if request.url.path in ("/login", "/logout"):
            return await call_next(request)

        cookie = request.cookies.get(AUTH_COOKIE_NAME, "")
        if cookie and hmac.compare_digest(cookie, self.expected_token):
            return await call_next(request)

        if request.url.path.startswith("/api/"):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
        return RedirectResponse(url="/login", status_code=303)


LOGIN_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<title>Вход — OKX Event-Contract Paper Bot</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; background:#0f1115; color:#e6e6e6;
         margin:0; height:100vh; display:flex; align-items:center; justify-content:center; }
  .box { background:#171a21; border:1px solid #23262e; border-radius:10px; padding:28px; width:280px; }
  h1 { font-size:15px; margin:0 0 16px; }
  input { width:100%; box-sizing:border-box; background:#0f1115; color:#e6e6e6; border:1px solid #2a2e37;
          border-radius:6px; padding:8px 10px; font-size:14px; margin-bottom:10px; }
  button { width:100%; background:#1f2e3a; color:#7fc7ff; border:1px solid #2b4a6b; border-radius:6px;
            padding:8px; font-size:14px; cursor:pointer; }
  button:hover { background:#26394a; }
  #err { color:#ff6b6b; font-size:12px; min-height:16px; margin-top:8px; }
</style>
</head>
<body>
  <div class="box">
    <h1>OKX Event-Contract Paper Bot</h1>
    <input type="password" id="pw" placeholder="Пароль" autofocus>
    <button onclick="doLogin()">Войти</button>
    <div id="err"></div>
  </div>
<script>
async function doLogin(){
  const err = document.getElementById('err');
  const password = document.getElementById('pw').value;
  try {
    const r = await fetch('/login', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ password }),
    });
    if (r.ok) { window.location.href = '/'; return; }
    err.textContent = 'Неверный пароль';
  } catch (e) {
    err.textContent = 'Ошибка сети: ' + e.message;
  }
}
document.getElementById('pw').addEventListener('keydown', e => { if (e.key === 'Enter') doLogin(); });
</script>
</body>
</html>
"""


class LoginRequest(BaseModel):
    password: str


class StrategySettingsItem(BaseModel):
    """One strategy's editable fields for POST /api/settings. All optional
    — only fields you actually send get changed; omitted ones keep their
    current value. `extra` is merged (not replaced) into the strategy's
    own extra config, so you only need to send the keys you're changing."""
    enabled: Optional[bool] = None
    deposit_usd: Optional[float] = None
    stake_fraction: Optional[float] = None       # fraction 0..1, e.g. 0.08 = 8%
    max_coefficient: Optional[float] = None
    entry_windows_min: Optional[list[float]] = None
    extra: Optional[dict[str, Any]] = None


class StrategiesSettings(BaseModel):
    strategies: dict[str, StrategySettingsItem]


class ResetStrategyRequest(BaseModel):
    strategy: str

INDEX_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<title>OKX Event-Contract Paper Bot</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; background:#0f1115; color:#e6e6e6; margin:0; padding:24px; }
  h1 { font-size:18px; margin:0 0 4px; }
  .sub { color:#9aa0a6; font-size:13px; margin-bottom:20px; }
  table { border-collapse: collapse; width:100%; margin-bottom:24px; }
  th, td { padding:6px 10px; text-align:right; border-bottom:1px solid #23262e; font-size:13px; }
  th:first-child, td:first-child { text-align:left; }
  th { color:#9aa0a6; font-weight:600; }
  .pos { color:#3ddc84; font-weight:600; }
  .neg { color:#ff6b6b; font-weight:600; }
  .card { background:#171a21; border:1px solid #23262e; border-radius:10px; padding:16px; margin-bottom:20px; }
  .leader { font-size:13px; line-height:1.9; }
  .badge { display:inline-block; padding:2px 8px; border-radius:6px; background:#23262e; font-size:11px; margin-left:8px; }
  .topbar { display:flex; align-items:baseline; justify-content:space-between; gap:12px; flex-wrap:wrap; }
  #resetBtn { background:#3a1f1f; color:#ff9f9f; border:1px solid #6b2b2b; border-radius:6px;
              padding:6px 12px; font-size:12px; cursor:pointer; }
  #resetBtn:hover { background:#4a2626; }
  #resetBtn:disabled { opacity:0.5; cursor:default; }
  #resetMsg { font-size:12px; color:#9aa0a6; margin-left:10px; }
  #logoutBtn { background:#171a21; color:#9aa0a6; border:1px solid #23262e; border-radius:6px;
               padding:6px 12px; font-size:12px; cursor:pointer; margin-right:6px; }
  #logoutBtn:hover { background:#1f232b; }
  .tabs { display:flex; gap:6px; margin:16px 0 20px; }
  .tab-btn { background:#171a21; color:#9aa0a6; border:1px solid #23262e; border-radius:6px;
             padding:6px 14px; font-size:13px; cursor:pointer; }
  .tab-btn.active { background:#1f2e3a; color:#7fc7ff; border-color:#2b4a6b; }
  .settings-row { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-bottom:10px; font-size:13px; }
  .settings-row label { display:flex; align-items:center; gap:6px; white-space:nowrap; }
  .settings-row input[type=number], .settings-row input[type=text] {
    width:90px; background:#0f1115; color:#e6e6e6; border:1px solid #2a2e37; border-radius:5px; padding:4px 6px; }
  .strategy-card h3 { display:flex; align-items:center; gap:12px; font-size:14px; margin:0 0 12px; }
  .strategy-card h3 label { font-size:12px; font-weight:normal; color:#9aa0a6; }
  .extra-row { border-top:1px dashed #23262e; padding-top:10px; }
  .extra-row input[type=text] { width:140px; }
  #applyStrategyBtn { background:#1f2e3a; color:#7fc7ff; border:1px solid #2b4a6b; border-radius:6px;
              padding:8px 16px; font-size:13px; cursor:pointer; margin-top:8px; }
  #applyStrategyBtn:hover { background:#26394a; }
  #strategySettingsMsg { font-size:12px; color:#9aa0a6; margin-left:10px; }
  .extra-row textarea { width:100%; min-height:70px; background:#0f1115; color:#e6e6e6;
    border:1px solid #2a2e37; border-radius:5px; padding:6px; font-family:inherit; font-size:12px;
    box-sizing:border-box; margin-top:4px; }
  .extra-field-block { display:flex; flex-direction:column; gap:4px; min-width:160px; }
  .reset-strategy-btn { background:#3a1f1f; color:#ff9f9f; border:1px solid #6b2b2b; border-radius:5px;
    padding:3px 10px; font-size:11px; cursor:pointer; margin-left:auto; }
  .reset-strategy-btn:hover { background:#4a2626; }
  .chart-card svg { width:100%; height:auto; display:block; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; margin-top:8px; }
  .legend-item { display:flex; align-items:center; gap:6px; }
  .legend-dot { width:10px; height:10px; border-radius:2px; display:inline-block; }
  .analytics-controls { display:flex; gap:12px; align-items:center; margin-bottom:12px; font-size:13px; flex-wrap:wrap; }
  .analytics-controls select { background:#0f1115; color:#e6e6e6; border:1px solid #2a2e37;
    border-radius:5px; padding:4px 8px; }
  #exportCsvBtn { background:#171a21; color:#7fc7ff; border:1px solid #2b4a6b; border-radius:6px;
    padding:5px 12px; font-size:12px; cursor:pointer; }
  #exportCsvBtn:hover { background:#1f2e3a; }
  .trades-stats { font-size:12px; color:#9aa0a6; margin-bottom:10px; }
  .table-scroll { overflow-x:auto; }
  #tradesTable th.sortable { cursor:pointer; user-select:none; white-space:nowrap; }
  #tradesTable th.sortable:hover { color:#e6e6e6; }
  #tradesTable th .sort-arrow { color:#7fc7ff; margin-left:3px; }
  .pager { display:flex; align-items:center; gap:10px; margin-top:12px; font-size:13px; }
  .pager button { background:#171a21; color:#9aa0a6; border:1px solid #23262e; border-radius:6px;
    padding:4px 12px; font-size:12px; cursor:pointer; }
  .pager button:hover:not(:disabled) { background:#1f232b; }
  .pager button:disabled { opacity:0.4; cursor:default; }
  .reason-cell { max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-align:left !important; }
  .live-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#3ddc84;
    margin-left:6px; vertical-align:middle; animation: live-pulse 1.6s ease-in-out infinite; }
  @keyframes live-pulse { 0%,100% { opacity:1; } 50% { opacity:0.25; } }
  #clearLogsBtn { background:#171a21; color:#9aa0a6; border:1px solid #23262e; border-radius:6px;
    padding:5px 12px; font-size:12px; cursor:pointer; }
  #clearLogsBtn:hover { background:#1f232b; }
  .activity-log { display:flex; flex-direction:column; gap:2px; max-height:70vh; overflow-y:auto;
    font-size:12.5px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .log-line { display:grid; grid-template-columns: 66px 108px 34px 84px 1fr; gap:8px; align-items:baseline;
    padding:4px 8px; border-radius:4px; border-left:3px solid transparent; }
  .log-line:nth-child(odd) { background:#14161c; }
  .log-time { color:#6b7280; white-space:nowrap; }
  .log-strategy { color:#9aa0a6; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .log-window { color:#6b7280; }
  .log-kind { font-weight:600; white-space:nowrap; }
  .log-msg { color:#c7cbd1; word-break:break-word; }
  .log-empty { color:#6b7280; padding:14px 8px; font-family:inherit; }
  /* entries/signals get a distinct, louder treatment than routine "no signal" checks */
  .log-no_signal { border-left-color:transparent; opacity:0.55; }
  .log-no_signal .log-kind { color:#6b7280; }
  .log-opened { border-left-color:#7fc7ff; background:rgba(127,199,255,0.08) !important; }
  .log-opened .log-kind { color:#7fc7ff; }
  .log-opened .log-msg { color:#e6e6e6; }
  .log-rejected { border-left-color:#ffb84d; }
  .log-rejected .log-kind { color:#ffb84d; }
  .log-won { border-left-color:#3ddc84; background:rgba(61,220,132,0.08) !important; }
  .log-won .log-kind { color:#3ddc84; }
  .log-won .log-msg { color:#e6e6e6; }
  .log-lost { border-left-color:#ff6b6b; background:rgba(255,107,107,0.08) !important; }
  .log-lost .log-kind { color:#ff6b6b; }
  .log-lost .log-msg { color:#e6e6e6; }
  .log-unresolved { border-left-color:#ffd166; }
  .log-unresolved .log-kind { color:#ffd166; }
  @media (max-width: 640px) {
    .log-line { grid-template-columns: 56px 1fr; grid-template-areas: "time strategy" "kind msg" "window window";
      row-gap:2px; }
    .log-time { grid-area:time; } .log-strategy { grid-area:strategy; }
    .log-window { grid-area:window; font-size:11px; }
    .log-kind { grid-area:kind; } .log-msg { grid-area:msg; }
  }
</style>
</head>
<body>
  <div class="topbar">
    <h1>OKX Event-Contract Paper Trading Bot <span id="mode" class="badge"></span></h1>
    <div>
      %LOGOUT_BTN%
      <button id="resetBtn" onclick="resetDb()">Сбросить базу (Reset DB)</button>
      <span id="resetMsg"></span>
    </div>
  </div>
  <div class="sub" id="meta">loading…</div>

  <div class="tabs">
    <button id="tabDashboardBtn" class="tab-btn active" onclick="showTab('dashboard')">Дашборд</button>
    <button id="tabAnalyticsBtn" class="tab-btn" onclick="showTab('analytics')">Аналитика</button>
    <button id="tabLogsBtn" class="tab-btn" onclick="showTab('logs')">Логи</button>
    <button id="tabSettingsBtn" class="tab-btn" onclick="showTab('settings')">Настройки стратегий</button>
  </div>

  <div id="dashboardView">
    <div class="card"><h3>Виртуальные кошельки</h3><table id="wallets"></table></div>
    <div class="card"><h3>Стратегия × время входа</h3><table id="combos"></table></div>
    <div class="card"><h3>Активные сделки</h3><table id="open"></table></div>
    <div class="card leader" id="leaderboard"></div>
  </div>

  <div id="analyticsView" style="display:none;">
    <div class="card chart-card"><h3>Кривая эквити по стратегиям</h3><div id="equityChart"></div></div>
    <div class="card chart-card"><h3>PnL по стратегиям, $</h3><div id="pnlChart"></div></div>
    <div class="card chart-card"><h3>Winrate по связкам «стратегия × окно», %</h3><div id="winrateChart"></div></div>
    <div class="card">
      <h3>История сделок</h3>
      <div class="analytics-controls">
        <label>Стратегия:
          <select id="tradesFilter" onchange="onTradesFilterChange()"><option value="">Все</option></select>
        </label>
        <button id="exportCsvBtn" onclick="exportTradesCsv()">⬇ Экспорт CSV</button>
      </div>
      <div id="tradesStats" class="trades-stats"></div>
      <div class="table-scroll"><table id="tradesTable"></table></div>
      <div class="pager" id="tradesPager"></div>
    </div>
  </div>

  <div id="logsView" style="display:none;">
    <div class="card">
      <h3>Логи стратегий <span id="logsLiveDot" class="live-dot" title="обновляется в реальном времени"></span></h3>
      <div class="analytics-controls">
        <label>Стратегия:
          <select id="logsFilter" onchange="renderActivityLog()"><option value="">Все</option></select>
        </label>
        <button id="clearLogsBtn" onclick="clearActivityView()">Очистить вид</button>
      </div>
      <div id="activityLog" class="activity-log">loading…</div>
    </div>
  </div>

  <div id="settingsView" style="display:none;">
    <div id="strategyCards">loading…</div>
    <button id="applyStrategyBtn" onclick="applyStrategySettings()">Применить (сброс всех данных)</button>
    <span id="strategySettingsMsg"></span>
  </div>

<script>
function money(v){ const s = v>0?'+':''; return `${s}$${v.toFixed(2)}`; }
function cls(v){ return v>0?'pos':(v<0?'neg':''); }
let currentTab = 'dashboard';
let strategySettingsLoaded = false;
let tradesFilterLoaded = false;
let logsFilterLoaded = false;
let activityEvents = [];      // client-side ring buffer, newest first
let activitySinceId = 0;
const ACTIVITY_MAX_BUFFER = 400;
const ACTIVITY_KIND_LABELS = {
  no_signal: 'нет сигнала', opened: 'ВХОД', rejected: 'отклонено',
  won: 'ПОБЕДА', lost: 'ПРОИГРЫШ', unresolved: 'не подтверждено',
};
let tradesPage = 0;
let tradesSortBy = 'closed_ts';
let tradesSortDir = 'desc';
const TRADES_PAGE_SIZE = 50;
const TRADE_COLUMNS = [
  { key: 'closed_ts', label: 'Закрыта' },
  { key: 'opened_ts', label: 'Открыта' },
  { key: 'strategy', label: 'Стратегия' },
  { key: 'entry_window_min', label: 'Окно' },
  { key: 'inst_id', label: 'Инструмент' },
  { key: 'direction', label: 'Напр.' },
  { key: 'entry_price', label: 'Вход' },
  { key: 'stake_usd', label: 'Стейк' },
  { key: 'duration_sec', label: 'Длит.' },
  { key: 'status', label: 'Статус' },
  { key: 'pnl_usd', label: 'PnL' },
  { key: 'roi_pct', label: 'ROI %' },
  { key: 'reason', label: 'Причина' },
];
const PALETTE = ['#7fc7ff', '#3ddc84', '#ffb84d', '#ff6b6b', '#c792ea', '#4dd0e1', '#f06292', '#a1887f'];

function showTab(tab){
  currentTab = tab;
  document.getElementById('dashboardView').style.display = tab === 'dashboard' ? '' : 'none';
  document.getElementById('analyticsView').style.display = tab === 'analytics' ? '' : 'none';
  document.getElementById('logsView').style.display = tab === 'logs' ? '' : 'none';
  document.getElementById('settingsView').style.display = tab === 'settings' ? '' : 'none';
  document.getElementById('tabDashboardBtn').classList.toggle('active', tab === 'dashboard');
  document.getElementById('tabAnalyticsBtn').classList.toggle('active', tab === 'analytics');
  document.getElementById('tabLogsBtn').classList.toggle('active', tab === 'logs');
  document.getElementById('tabSettingsBtn').classList.toggle('active', tab === 'settings');
  if (tab === 'settings' && !strategySettingsLoaded) loadStrategySettings();
  if (tab === 'analytics') {
    if (!tradesFilterLoaded) loadTradesFilterOptions();
    loadTradesTable();
  }
  if (tab === 'logs') {
    if (!logsFilterLoaded) loadLogsFilterOptions();
    renderActivityLog();
  }
}

async function loadLogsFilterOptions(){
  const r = await fetch('/api/strategies');
  const strategies = await r.json();
  const sel = document.getElementById('logsFilter');
  for (const s of strategies) {
    const opt = document.createElement('option');
    opt.value = s.name;
    opt.textContent = s.display_name;
    sel.appendChild(opt);
  }
  logsFilterLoaded = true;
}

// Polled from tick() every refresh cycle regardless of which tab is
// active — the payload is tiny (only events newer than activitySinceId)
// so the log tab has data ready the instant you switch to it, instead of
// starting from empty.
async function loadActivity(){
  const r = await fetch(`/api/activity?since_id=${activitySinceId}`);
  const d = await r.json();
  if (d.events && d.events.length) {
    activityEvents = d.events.slice().reverse().concat(activityEvents).slice(0, ACTIVITY_MAX_BUFFER);
  }
  activitySinceId = d.latest_id;
  if (currentTab === 'logs') renderActivityLog();
}

function renderActivityLog(){
  const filter = document.getElementById('logsFilter').value;
  const rows = filter ? activityEvents.filter(e => e.strategy === filter) : activityEvents;
  if (!rows.length) {
    document.getElementById('activityLog').innerHTML =
      '<div class="log-empty">пока пусто — ждём следующего чекпоинта входа</div>';
    return;
  }
  const html = rows.map(e => {
    const t = new Date(e.ts * 1000).toLocaleTimeString();
    const win = e.window_min != null ? `${e.window_min}м` : '';
    const label = ACTIVITY_KIND_LABELS[e.kind] || e.kind;
    return `<div class="log-line log-${e.kind}">
      <span class="log-time">${t}</span>
      <span class="log-strategy">${escapeHtml(e.display_name)}</span>
      <span class="log-window">${win}</span>
      <span class="log-kind">${label}</span>
      <span class="log-msg">${escapeHtml(e.message)}</span>
    </div>`;
  }).join('');
  document.getElementById('activityLog').innerHTML = html;
}

function clearActivityView(){
  // Clears only what's rendered client-side — a fresh poll will still
  // pull in anything new; this is "tidy the screen", not "wipe history"
  // (that's the Reset DB button, a different, destructive action).
  activityEvents = [];
  renderActivityLog();
}

async function loadTradesFilterOptions(){
  const r = await fetch('/api/strategies');
  const strategies = await r.json();
  const sel = document.getElementById('tradesFilter');
  for (const s of strategies) {
    const opt = document.createElement('option');
    opt.value = s.name;
    opt.textContent = s.display_name;
    sel.appendChild(opt);
  }
  tradesFilterLoaded = true;
}

function onTradesFilterChange(){
  tradesPage = 0;
  loadTradesTable();
}

function onTradesSort(key){
  if (tradesSortBy === key) {
    tradesSortDir = tradesSortDir === 'asc' ? 'desc' : 'asc';
  } else {
    tradesSortBy = key;
    tradesSortDir = 'desc';
  }
  tradesPage = 0;
  loadTradesTable();
}

function tradesGoPage(page){
  tradesPage = Math.max(0, page);
  loadTradesTable();
}

function fmtDuration(sec){
  if (sec == null || isNaN(sec)) return '—';
  sec = Math.max(0, Math.round(sec));
  return `${Math.floor(sec / 60)}м ${sec % 60}с`;
}

function tradesHeaderRow(){
  return '<tr>' + TRADE_COLUMNS.map(c => {
    const arrow = tradesSortBy === c.key ? `<span class="sort-arrow">${tradesSortDir === 'asc' ? '▲' : '▼'}</span>` : '';
    return `<th class="sortable" onclick="onTradesSort('${c.key}')">${c.label}${arrow}</th>`;
  }).join('') + '</tr>';
}

function renderTradesPager(total){
  const totalPages = Math.max(1, Math.ceil(total / TRADES_PAGE_SIZE));
  const page = tradesPage + 1;
  document.getElementById('tradesPager').innerHTML = `
    <button onclick="tradesGoPage(0)" ${tradesPage === 0 ? 'disabled' : ''}>« Первая</button>
    <button onclick="tradesGoPage(${tradesPage - 1})" ${tradesPage === 0 ? 'disabled' : ''}>‹ Пред.</button>
    <span>Стр. ${page} из ${totalPages} (${total} сделок)</span>
    <button onclick="tradesGoPage(${tradesPage + 1})" ${page >= totalPages ? 'disabled' : ''}>След. ›</button>
    <button onclick="tradesGoPage(${totalPages - 1})" ${page >= totalPages ? 'disabled' : ''}>Последняя »</button>
  `;
}

function exportTradesCsv(){
  const strategy = document.getElementById('tradesFilter').value;
  const params = new URLSearchParams({ sort_by: tradesSortBy, sort_dir: tradesSortDir });
  if (strategy) params.set('strategy', strategy);
  window.location.href = '/api/trades/export.csv?' + params.toString();
}

async function loadTradesTable(){
  const strategy = document.getElementById('tradesFilter').value;
  const params = new URLSearchParams({
    limit: TRADES_PAGE_SIZE, offset: tradesPage * TRADES_PAGE_SIZE,
    sort_by: tradesSortBy, sort_dir: tradesSortDir,
  });
  if (strategy) params.set('strategy', strategy);
  const r = await fetch('/api/trades?' + params.toString());
  const d = await r.json();

  let html = tradesHeaderRow();
  for (const t of d.rows) {
    const closedAt = t.closed_ts ? new Date(t.closed_ts * 1000).toLocaleString() : '—';
    const openedAt = t.opened_ts ? new Date(t.opened_ts * 1000).toLocaleString() : '—';
    const pnl = t.pnl_usd ?? 0;
    const roi = t.stake_usd > 0 ? (pnl / t.stake_usd * 100) : 0;
    const duration = (t.closed_ts && t.opened_ts) ? t.closed_ts - t.opened_ts : null;
    html += `<tr>
      <td>${closedAt}</td><td>${openedAt}</td><td>${t.strategy}</td><td>${t.entry_window_min} мин</td>
      <td>${t.inst_id}</td><td>${t.direction}</td><td>$${Number(t.entry_price).toFixed(3)}</td>
      <td>$${Number(t.stake_usd).toFixed(2)}</td><td>${fmtDuration(duration)}</td><td>${t.status}</td>
      <td class="${cls(pnl)}">${money(pnl)}</td><td class="${cls(roi)}">${roi.toFixed(1)}%</td>
      <td class="reason-cell" title="${escapeHtml(t.reason || '')}">${escapeHtml(t.reason || '—')}</td>
    </tr>`;
  }
  document.getElementById('tradesTable').innerHTML = html;

  const st = d.stats;
  document.getElementById('tradesStats').textContent = st.total
    ? `Всего: ${st.total} сделок (${st.wins}W/${st.losses}L, winrate ${st.winrate_pct.toFixed(1)}%) — суммарный PnL: ${money(st.net_pnl)}`
    : 'Нет закрытых сделок по этому фильтру';

  renderTradesPager(d.total);
}

// -- tiny inline-SVG charts (no charting library) ----------------------------------
function svgLineChart(series, width, height){
  const pad = 34;
  let allPoints = series.flatMap(s => s.points);
  if (allPoints.length === 0) return '<svg viewBox="0 0 ' + width + ' ' + height + '"></svg>';
  const tMin = Math.min(...allPoints.map(p => p.t)), tMax = Math.max(...allPoints.map(p => p.t));
  const eMin = Math.min(...allPoints.map(p => p.equity)), eMax = Math.max(...allPoints.map(p => p.equity));
  const eSpan = (eMax - eMin) || 1, tSpan = (tMax - tMin) || 1;
  const x = t => pad + (t - tMin) / tSpan * (width - 2 * pad);
  const y = e => height - pad - (e - eMin) / eSpan * (height - 2 * pad);

  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<line x1="${pad}" y1="${height-pad}" x2="${width-pad}" y2="${height-pad}" stroke="#2a2e37"/>`;
  svg += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height-pad}" stroke="#2a2e37"/>`;
  svg += `<text x="4" y="${pad+4}" fill="#9aa0a6" font-size="10">$${eMax.toFixed(0)}</text>`;
  svg += `<text x="4" y="${height-pad}" fill="#9aa0a6" font-size="10">$${eMin.toFixed(0)}</text>`;
  series.forEach((s, i) => {
    if (s.points.length === 0) return;
    const d = s.points.map(p => `${x(p.t)},${y(p.equity)}`).join(' ');
    svg += `<polyline points="${d}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
  });
  svg += '</svg>';
  return svg;
}

function svgBarChart(items, width, height){
  const pad = 34;
  if (items.length === 0) return `<svg viewBox="0 0 ${width} ${height}"></svg>`;
  const vMax = Math.max(1, ...items.map(it => Math.abs(it.value)));
  const zeroY = height - pad;
  const barW = (width - 2 * pad) / items.length;
  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<line x1="${pad}" y1="${zeroY}" x2="${width-pad}" y2="${zeroY}" stroke="#2a2e37"/>`;
  items.forEach((it, i) => {
    const h = Math.abs(it.value) / vMax * (height - 2 * pad - 20);
    const barX = pad + i * barW + barW * 0.15;
    const barWidth = barW * 0.7;
    const barY = it.value >= 0 ? zeroY - h : zeroY;
    svg += `<rect x="${barX}" y="${barY}" width="${barWidth}" height="${Math.max(h,1)}" fill="${it.color}"/>`;
    svg += `<text x="${barX + barWidth/2}" y="${zeroY + 14}" fill="#9aa0a6" font-size="9" text-anchor="middle">` +
           `${it.label.length > 10 ? it.label.slice(0,10)+'…' : it.label}</text>`;
    svg += `<text x="${barX + barWidth/2}" y="${it.value >= 0 ? barY - 4 : barY + h + 12}" fill="#e6e6e6" ` +
           `font-size="9" text-anchor="middle">${it.value.toFixed(1)}</text>`;
  });
  svg += '</svg>';
  return svg;
}

function renderAnalytics(d){
  const series = d.wallets.map((w, i) => ({
    name: w.display_name, color: PALETTE[i % PALETTE.length], points: w.equity_curve || [],
  }));
  document.getElementById('equityChart').innerHTML = svgLineChart(series, 760, 260);
  const legend = series.map(s =>
    `<div class="legend-item"><span class="legend-dot" style="background:${s.color}"></span>${s.name}</div>`
  ).join('');
  document.getElementById('equityChart').innerHTML += `<div class="legend">${legend}</div>`;

  const pnlItems = d.wallets.map((w, i) => ({
    label: w.display_name, value: w.net_pnl, color: w.net_pnl >= 0 ? '#3ddc84' : '#ff6b6b',
  }));
  document.getElementById('pnlChart').innerHTML = svgBarChart(pnlItems, 760, 220);

  const winrateItems = d.combos.map((c, i) => ({
    label: `${c.display_name} @${c.window_min}м`, value: c.winrate_pct, color: PALETTE[i % PALETTE.length],
  }));
  document.getElementById('winrateChart').innerHTML = svgBarChart(winrateItems, 760, 220);
}

async function loadStrategySettings(){
  const r = await fetch('/api/strategies');
  renderStrategySettings(await r.json());
  strategySettingsLoaded = true;
}

function escapeHtml(v){
  return String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderStrategySettings(strategies){
  const root = document.getElementById('strategyCards');
  root.innerHTML = '';
  for (const s of strategies) {
    let extraHtml = '';
    for (const [k, v] of Object.entries(s.extra || {})) {
      const isNum = typeof v === 'number';
      const isLongText = !isNum && (String(v).includes('\\n') || String(v).length > 50);
      if (isLongText) {
        extraHtml += `<div class="extra-field-block"><label>${k}</label>
          <textarea data-strategy="${s.name}" data-extra-key="${k}">${escapeHtml(v)}</textarea></div>`;
      } else {
        extraHtml += `<label>${k} <input type="${isNum ? 'number' : 'text'}" step="any"
          value="${escapeHtml(v)}" data-strategy="${s.name}" data-extra-key="${k}"></label>`;
      }
    }
    const div = document.createElement('div');
    div.className = 'card strategy-card';
    div.innerHTML = `
      <h3>${s.display_name}
        <label><input type="checkbox" data-strategy="${s.name}" data-field="enabled"
                      ${s.enabled ? 'checked' : ''}> включена</label>
        <button class="reset-strategy-btn" onclick="resetOneStrategy('${s.name}')">Сбросить эту стратегию</button>
      </h3>
      <div class="settings-row">
        <label>Депозит, $ <input type="number" min="1" step="1" value="${s.deposit_usd}"
                                  data-strategy="${s.name}" data-field="deposit_usd"></label>
        <label>Ставка от баланса, % <input type="number" min="0.1" max="100" step="0.1"
                                  value="${(s.stake_fraction * 100).toFixed(2)}"
                                  data-strategy="${s.name}" data-field="stake_fraction_pct"></label>
        <label>Макс. коэфф. входа <input type="number" min="0.01" max="1" step="0.01"
                                  value="${s.max_coefficient}"
                                  data-strategy="${s.name}" data-field="max_coefficient"></label>
        <label>Окна входа (мин, через запятую) <input type="text" style="width:140px;"
                                  value="${s.entry_windows_min.join(',')}"
                                  data-strategy="${s.name}" data-field="entry_windows_min"></label>
      </div>
      ${extraHtml ? `<div class="settings-row extra-row">${extraHtml}</div>` : ''}
    `;
    root.appendChild(div);
  }
}

async function applyStrategySettings(){
  const msg = document.getElementById('strategySettingsMsg');
  const byName = {};
  const ensure = (name) => byName[name] || (byName[name] = { extra: {} });

  document.querySelectorAll('#strategyCards input[data-strategy], #strategyCards textarea[data-strategy]').forEach(inp => {
    const entry = ensure(inp.dataset.strategy);
    if (inp.dataset.extraKey) {
      entry.extra[inp.dataset.extraKey] = inp.type === 'number' ? parseFloat(inp.value) : inp.value;
      return;
    }
    const field = inp.dataset.field;
    if (field === 'enabled') { entry.enabled = inp.checked; return; }
    if (field === 'entry_windows_min') {
      entry.entry_windows_min = inp.value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
      return;
    }
    if (field === 'stake_fraction_pct') { entry.stake_fraction = parseFloat(inp.value) / 100; return; }
    entry[field] = parseFloat(inp.value);
  });
  for (const name in byName) {
    if (Object.keys(byName[name].extra).length === 0) delete byName[name].extra;
  }

  if (!confirm('Настройки стратегий изменятся и вся история сделок будет сброшена (как ' +
               'кнопка "Сбросить базу"). Продолжить?')) return;

  msg.textContent = 'применяю…';
  try {
    const r = await fetch('/api/settings', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ strategies: byName }),
    });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || ('HTTP ' + r.status));
    msg.textContent = 'применено ✓';
    strategySettingsLoaded = false;
    if (currentTab === 'settings') await loadStrategySettings();
    await tick();
  } catch (e) {
    msg.textContent = 'ошибка: ' + e.message;
  }
  setTimeout(() => { msg.textContent = ''; }, 5000);
}

async function tick(){
  loadActivity();  // fire-and-forget — polled every tick regardless of active tab, see its own docstring
  const r = await fetch('/api/state');
  const d = await r.json();

  document.getElementById('mode').textContent = d.mock_mode ? 'MOCK DATA' : 'OKX DEMO TRADING';
  document.getElementById('meta').textContent =
    `BTC/USDT: ${d.underlying_price ?? '—'}   updated ${new Date(d.updated_at*1000).toLocaleTimeString()}`;

  let w = '<tr><th>Стратегия</th><th>Депозит</th><th>Баланс</th><th>В сделках</th><th>Equity</th><th>PnL</th><th>Открыто</th></tr>';
  for (const x of d.wallets) {
    w += `<tr><td>${x.display_name}</td><td>$${x.initial_balance.toFixed(2)}</td><td>$${x.balance.toFixed(2)}</td>
          <td>$${x.reserved.toFixed(2)}</td><td>$${x.equity.toFixed(2)}</td>
          <td class="${cls(x.net_pnl)}">${money(x.net_pnl)}</td><td>${x.open_trades}</td></tr>`;
  }
  document.getElementById('wallets').innerHTML = w;

  let c = '<tr><th>Стратегия</th><th>Окно</th><th>Сделок</th><th>Winrate</th><th>Ср.коэфф</th><th>PnL</th><th>ROI</th></tr>';
  for (const x of d.combos) {
    c += `<tr><td>${x.display_name}</td><td>${x.window_min} мин</td><td>${x.trades} (${x.wins}W/${x.losses}L)</td>
          <td>${x.winrate_pct.toFixed(1)}%</td><td>$${x.avg_entry_price.toFixed(3)}</td>
          <td class="${cls(x.net_pnl)}">${money(x.net_pnl)}</td>
          <td class="${cls(x.roi_pct)}">${x.roi_pct.toFixed(1)}%</td></tr>`;
  }
  document.getElementById('combos').innerHTML = c || '';

  let o = '<tr><th>Стратегия</th><th>Инструмент</th><th>Напр.</th><th>Вход</th><th>Стейк</th><th>Осталось</th></tr>';
  for (const x of d.open_trades) {
    o += `<tr><td>${x.strategy}</td><td>${x.inst_id}</td><td>${x.direction}</td>
          <td>$${x.entry_price.toFixed(3)}</td><td>$${x.stake_usd.toFixed(2)}</td><td>${x.remaining_sec}s</td></tr>`;
  }
  document.getElementById('open').innerHTML = o;

  const lb = d.leaderboard;
  function line(label, x){
    if(!x) return `${label}: недостаточно данных<br>`;
    return `${label}: <b>${x.display_name}</b> @ ${x.window_min} мин — winrate ${x.winrate_pct.toFixed(1)}%, `
         + `ср.коэфф $${x.avg_entry_price.toFixed(3)}, PnL ${money(x.net_pnl)} (${x.trades} сделок)<br>`;
  }
  document.getElementById('leaderboard').innerHTML =
    '<h3>🏆 Лидерборд</h3>' + line('Лучшая связка по PnL', lb.best_pnl)
    + line('Лучший Winrate', lb.best_winrate) + line('Лучшая ROI', lb.best_value);

  if (currentTab === 'analytics') renderAnalytics(d);
}

async function resetOneStrategy(name){
  if (!confirm(`Сбросить только "${name}"? Её кошелёк и история сделок начнутся заново, ` +
               'остальные стратегии не тронет. Действие необратимо.')) return;
  try {
    const r = await fetch('/api/reset_strategy', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ strategy: name }),
    });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || ('HTTP ' + r.status));
    strategySettingsLoaded = false;
    if (currentTab === 'settings') await loadStrategySettings();
    await tick();
  } catch (e) {
    alert('Ошибка сброса: ' + e.message);
  }
}

async function logout(){
  await fetch('/logout', { method: 'POST' });
  window.location.href = '/login';
}

async function resetDb(){
  if(!confirm('Сбросить базу? Все виртуальные кошельки, сделки и история (data/bot.db) будут ' +
              'стёрты и стратегии начнут заново со своим текущим депозитом. Действие необратимо.')) return;
  const btn = document.getElementById('resetBtn');
  const msg = document.getElementById('resetMsg');
  btn.disabled = true;
  msg.textContent = 'сбрасываю…';
  try {
    const r = await fetch('/api/reset', { method: 'POST' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    msg.textContent = 'готово ✓';
    await tick();
  } catch (e) {
    msg.textContent = 'ошибка: ' + e.message;
  } finally {
    btn.disabled = false;
    setTimeout(() => { msg.textContent = ''; }, 4000);
  }
}

tick();
setInterval(tick, %REFRESH_MS%);
</script>
</body>
</html>
"""


def _equity_curve(wallet, max_points: int = 300) -> list[dict]:
    """Reconstruct a (timestamp, equity) series from a wallet's closed
    trades — no separate snapshot table needed. Downsampled to
    `max_points` (keeping the very last point) so a long-running strategy
    doesn't bloat the /api/state payload every poll."""
    closed = sorted((t for t in wallet.trades if t.closed_ts is not None), key=lambda t: t.closed_ts)
    if not closed:
        return [{"t": time.time(), "equity": wallet.initial_balance}]

    points = [{"t": closed[0].opened_ts, "equity": wallet.initial_balance}]
    equity = wallet.initial_balance
    for t in closed:
        equity += t.pnl_usd or 0.0
        points.append({"t": t.closed_ts, "equity": equity})

    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        sampled = points[::step]
        if sampled[-1] is not points[-1]:
            sampled.append(points[-1])
        points = sampled
    return points


def build_app(cfg: AppConfig, engine: Engine) -> FastAPI:
    app = FastAPI(title="OKX Event-Contract Paper Bot")
    display_names = {s.name: s.display_name for s in cfg.strategies}

    password = os.getenv("DASHBOARD_PASSWORD", "").strip()
    app.add_middleware(AuthMiddleware, password=password)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page() -> str:
        return LOGIN_HTML

    @app.post("/login")
    async def login_submit(body: LoginRequest) -> JSONResponse:
        if not password or not hmac.compare_digest(body.password, password):
            return JSONResponse({"ok": False, "error": "wrong password"}, status_code=401)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(
            AUTH_COOKIE_NAME, _session_token(password), max_age=AUTH_COOKIE_MAX_AGE,
            httponly=True, samesite="lax",
        )
        return resp

    @app.post("/logout")
    async def logout() -> Response:
        resp = RedirectResponse(url="/login", status_code=303)
        resp.delete_cookie(AUTH_COOKIE_NAME)
        return resp

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        html = INDEX_HTML.replace("%REFRESH_MS%", str(int(cfg.dashboard.refresh_sec * 1000)))
        html = html.replace("%LOGOUT_BTN%", '<button id="logoutBtn" onclick="logout()">Выйти</button>' if password else "")
        return html

    @app.post("/api/reset")
    async def reset() -> JSONResponse:
        """Wipe all wallets/trades/history and start every strategy fresh
        from its current deposit. Local paper-trading state only — never
        touches a real OKX account."""
        engine.reset()
        return JSONResponse({"ok": True, "reset_at": time.time()})

    @app.post("/api/reset_strategy")
    async def reset_strategy(body: ResetStrategyRequest) -> JSONResponse:
        """Wipe ONE strategy's wallet/trades/history only — every other
        strategy's wallet and persisted rows are left untouched. Local
        paper-trading state only."""
        try:
            engine.reset_strategy(body.strategy)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "reset_at": time.time()})

    @app.get("/api/strategies")
    async def get_strategies() -> JSONResponse:
        """Full editable config for every strategy — backs the Settings tab."""
        return JSONResponse(
            [
                {
                    "name": s.name,
                    "display_name": s.display_name,
                    "enabled": s.enabled,
                    "deposit_usd": s.deposit_usd,
                    "stake_fraction": s.stake_fraction,
                    "max_coefficient": s.max_coefficient,
                    "entry_windows_min": s.entry_windows_min,
                    "extra": s.extra,
                }
                for s in cfg.strategies
            ]
        )

    @app.post("/api/settings")
    async def update_settings(body: StrategiesSettings) -> JSONResponse:
        """Apply partial per-strategy setting changes from the Settings
        tab. Always resets (like /api/reset) — there's no coherent way to
        swap a strategy's behavior or resize its wallet mid-trade, so
        every strategy restarts fresh. Local paper-trading state only."""
        if not body.strategies:
            return JSONResponse({"ok": False, "error": "strategies must not be empty"}, status_code=400)
        updates = {name: item.model_dump(exclude_unset=True) for name, item in body.strategies.items()}
        try:
            engine.update_strategy_settings(updates)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True})

    @app.get("/api/state")
    async def state() -> JSONResponse:
        snap = engine.snapshot()

        wallets = [
            {
                "strategy": name,
                "display_name": display_names.get(name, name),
                "initial_balance": w.initial_balance,
                "balance": w.balance,
                "reserved": w.reserved,
                "equity": w.equity,
                "net_pnl": w.net_pnl,
                "open_trades": len(w.open_trades()),
                "equity_curve": _equity_curve(w),
            }
            for name, w in snap.wallets.items()
        ]

        combos = [
            {
                "strategy": c.strategy,
                "display_name": display_names.get(c.strategy, c.strategy),
                "window_min": c.window_min,
                "trades": c.trades,
                "wins": c.wins,
                "losses": c.losses,
                "winrate_pct": c.winrate_pct,
                "avg_entry_price": c.avg_entry_price,
                "net_pnl": c.net_pnl,
                "roi_pct": c.roi_pct,
            }
            for c in sorted(snap.combo_stats.values(), key=lambda c: (c.strategy, -c.window_min))
        ]

        now = time.time()
        open_trades = [
            {
                "strategy": t.strategy, "inst_id": t.inst_id, "direction": t.direction.value.upper(),
                "entry_price": t.entry_price, "stake_usd": t.stake_usd,
                "remaining_sec": max(0, int(t.expiry_ts - now)),
            }
            for w in snap.wallets.values() for t in w.open_trades()
        ]

        def combo_json(c):
            if c is None:
                return None
            return {
                "display_name": display_names.get(c.strategy, c.strategy), "window_min": c.window_min,
                "winrate_pct": c.winrate_pct, "avg_entry_price": c.avg_entry_price,
                "net_pnl": c.net_pnl, "trades": c.trades,
            }

        return JSONResponse(
            {
                "mock_mode": cfg.mock_mode,
                "underlying_price": snap.underlying_price,
                "updated_at": snap.updated_at,
                "wallets": wallets,
                "combos": combos,
                "open_trades": open_trades,
                "leaderboard": {
                    "best_pnl": combo_json(snap.leaderboard.best_pnl),
                    "best_winrate": combo_json(snap.leaderboard.best_winrate),
                    "best_value": combo_json(snap.leaderboard.best_value),
                },
            }
        )

    @app.get("/api/trades")
    async def get_trades(
        strategy: Optional[str] = None, limit: int = 50, offset: int = 0,
        sort_by: str = "closed_ts", sort_dir: str = "desc",
    ) -> JSONResponse:
        """Paginated, sortable closed-trade history for the Analytics tab's
        table — reads from the durable SQLite log (data/bot.db), not just
        this process's in-memory state, so it also reflects rows from
        before the last restart. `total`/`stats` are computed over every
        row matching the filter, not just the current page, so the
        pager and the summary line stay accurate regardless of page size."""
        limit = min(max(limit, 1), 1000)
        offset = max(offset, 0)
        rows = engine.storage.get_trades(
            strategy=strategy, limit=limit, offset=offset, sort_by=sort_by, sort_dir=sort_dir
        )
        return JSONResponse(
            {
                "rows": rows,
                "total": engine.storage.count_trades(strategy=strategy),
                "stats": engine.storage.trades_stats(strategy=strategy),
            }
        )

    @app.get("/api/trades/export.csv")
    async def export_trades_csv(
        strategy: Optional[str] = None, sort_by: str = "closed_ts", sort_dir: str = "desc",
    ) -> StreamingResponse:
        """Full closed-trade history (every row matching the filter, not
        just one page) as a downloadable CSV — same filter/sort as the
        on-screen table, but unpaginated."""
        rows = engine.storage.get_trades(strategy=strategy, limit=None, sort_by=sort_by, sort_dir=sort_dir)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=TRADE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        filename = f"trades_{strategy or 'all'}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            iter([buf.getvalue()]), media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/activity")
    async def get_activity(since_id: int = 0, strategy: Optional[str] = None, limit: int = 300) -> JSONResponse:
        """Live per-strategy activity feed for the Dashboard's "Логи" tab —
        every strategy evaluation at a due entry-window checkpoint (signal
        or none), every rejection reason, and every settlement, as they
        happen. In-memory only (Engine._activity, not SQLite) — poll with
        `since_id` set to the previous response's `latest_id` to fetch just
        what's new instead of re-sending the whole buffer every time."""
        limit = min(max(limit, 1), 1000)
        events = engine.activity_since(since_id=since_id, strategy=strategy, limit=limit)
        return JSONResponse(
            {
                "events": [
                    {
                        "id": e.id, "ts": e.ts, "strategy": e.strategy,
                        "display_name": display_names.get(e.strategy, e.strategy),
                        "series_id": e.series_id, "window_min": e.window_min,
                        "kind": e.kind, "message": e.message,
                    }
                    for e in events
                ],
                "latest_id": engine.latest_activity_id(),
            }
        )

    return app
