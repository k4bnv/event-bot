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
from .models import TradeStatus
from .storage import FEATURE_FIELDS, TRADE_FIELDS

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
  #resetBtn, #resetFeaturesBtn { background:#3a1f1f; color:#ff9f9f; border:1px solid #6b2b2b; border-radius:6px;
              padding:6px 12px; font-size:12px; cursor:pointer; }
  #resetBtn:hover, #resetFeaturesBtn:hover { background:#4a2626; }
  #resetBtn:disabled, #resetFeaturesBtn:disabled { opacity:0.5; cursor:default; }
  #resetMsg, #resetFeaturesMsg { font-size:12px; color:#9aa0a6; margin-left:10px; }
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
  .chart-note { font-size:12px; color:#898781; margin-top:6px; }
  .analytics-controls { display:flex; gap:12px; align-items:center; margin-bottom:12px; font-size:13px; flex-wrap:wrap; }
  .analytics-controls select { background:#0f1115; color:#e6e6e6; border:1px solid #2a2e37;
    border-radius:5px; padding:4px 8px; }
  #exportCsvBtn, #exportFeaturesCsvBtn { background:#171a21; color:#7fc7ff; border:1px solid #2b4a6b;
    border-radius:6px; padding:5px 12px; font-size:12px; cursor:pointer; }
  #exportCsvBtn:hover, #exportFeaturesCsvBtn:hover { background:#1f2e3a; }
  .trades-stats { font-size:12px; color:#9aa0a6; margin-bottom:10px; }
  .hbar-row:hover rect { filter:brightness(1.15); }
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
  /* Диагностика tab — a strategy's decision funnel as a 100%-stacked bar
     (see renderDiagnostics): status-style semantic colors (good/quiet/
     fixable-blocked/other-blocked), not the categorical identity PALETTE
     — these four ARE state, not series identity. */
  #diagTable td.diag-bar-cell { text-align:left; min-width:200px; }
  .diag-stackbar { display:flex; height:16px; border-radius:4px; overflow:hidden; background:#0f1115; }
  .diag-stackbar .seg { height:100%; }
  .diag-stackbar .seg + .seg { margin-left:2px; }
  .diag-price-stats { font-size:11px; color:#898781; white-space:nowrap; text-align:left !important; }
  #diagRefreshBtn { background:#171a21; color:#7fc7ff; border:1px solid #2b4a6b; border-radius:6px;
    padding:5px 12px; font-size:12px; cursor:pointer; }
  #diagRefreshBtn:hover { background:#1f2e3a; }
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
    <button id="tabDiagnosticsBtn" class="tab-btn" onclick="showTab('diagnostics')">Диагностика</button>
  </div>

  <div id="dashboardView">
    <div class="card"><h3>Виртуальные кошельки</h3><table id="wallets"></table></div>
    <div class="card"><h3>Активные сделки</h3><table id="open"></table></div>
    <div class="card leader" id="leaderboard"></div>
  </div>

  <div id="analyticsView" style="display:none;">
    <div class="card chart-card">
      <h3>Кривая эквити по стратегиям, % от старта</h3>
      <div id="equityChart"></div>
      <div id="equityNote" class="chart-note"></div>
    </div>
    <div class="card chart-card"><h3>PnL по стратегиям, $</h3><div id="pnlChart"></div></div>
    <div class="card chart-card"><h3>Winrate по связкам «стратегия × окно», %</h3><div id="winrateChart"></div></div>
    <div class="card chart-card"><h3>По часу входа (UTC)</h3><div id="byHourChart"></div></div>
    <div class="card"><h3>По волатильности рынка при входе</h3><table id="byVolatility"></table></div>
    <div class="card"><h3>По направлению тренда при входе</h3><table id="byTrend"></table></div>
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
    <div class="card">
      <h3>Данные для обучения (ML)</h3>
      <p class="chart-note">Полный лог каждой оценки чекпоинта (в т.ч. когда сигнала не было) —
        см. таблицу checkpoint_features. Этот лог НЕ чистится обычным «Сбросить базу» (см. вкладку
        Диагностика) — специально, чтобы обучающие данные не терялись при каждой перенастройке.
        Строк: <span id="featuresCount">…</span></p>
      <div class="analytics-controls">
        <button id="exportFeaturesCsvBtn" onclick="exportFeaturesCsv()">⬇ Экспорт CSV для обучения</button>
        <button id="resetFeaturesBtn" onclick="resetFeatures()">🗑 Сбросить ML-данные</button>
        <span id="resetFeaturesMsg"></span>
      </div>
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

  <div id="diagnosticsView" style="display:none;">
    <div class="card">
      <h3>Почему стратегия (не) торгует</h3>
      <p class="chart-note">Разбор каждой оценки чекпоинта по каждой стратегии — то же самое, что
        scripts/decision_breakdown.py печатает из терминала, только без SSH. «Без сигнала» — своя
        логика входа стратегии ничего не нашла; «отклонено по цене» — сигнал был, но контракт стоил
        дороже max_coefficient (можно решить, подняв потолок — см. цены справа); «прочий отказ» —
        технические причины (нет котировки/баланса/проскальзывание). Этот лог НЕ чистится при сбросе
        базы — «За всё время» может смешивать историю до сброса/смены логики стратегии с текущей;
        выбери период поуже, чтобы увидеть, что происходит прямо сейчас.</p>
      <div class="analytics-controls">
        <label>Период:
          <select id="diagPeriod" onchange="loadDiagnostics()">
            <option value="">За всё время</option>
            <option value="1">Последний час</option>
            <option value="6" selected>Последние 6 часов</option>
            <option value="24">Последние 24 часа</option>
          </select>
        </label>
        <button id="diagRefreshBtn" onclick="loadDiagnostics()">↻ Обновить</button>
      </div>
      <div class="legend" id="diagLegend"></div>
      <div class="table-scroll"><table id="diagTable"></table></div>
    </div>
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
// Validated categorical palette (dark-surface steps) — CVD-safe adjacent
// order for stacks/bars/lines, up to 8 series (see dataviz skill /
// references/palette.md). Past 8 series, fold the tail rather than
// generate a 9th hue — see renderAnalytics' equity chart.
const PALETTE = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
// Status pair — matches the .pos/.neg text color already used everywhere
// else on this page (wallets/trades tables), so a chart's green/red means
// the same thing as the rest of the dashboard.
const STATUS_GOOD = '#3ddc84', STATUS_BAD = '#ff6b6b';
// Sequential single-hue (blue) ramp for a 0..1 magnitude on a DARK chart
// surface — brighter = higher, muted-toward-surface = lower (the dark-mode
// mirror of "more is darker" on a light surface).
function sequentialBlue(t){
  t = Math.max(0, Math.min(1, t));
  const lo = [24, 60, 100], hi = [140, 190, 245];   // muted navy -> bright blue
  const mix = (a, b) => Math.round(a + (b - a) * t);
  return `rgb(${mix(lo[0],hi[0])},${mix(lo[1],hi[1])},${mix(lo[2],hi[2])})`;
}

function showTab(tab){
  currentTab = tab;
  document.getElementById('dashboardView').style.display = tab === 'dashboard' ? '' : 'none';
  document.getElementById('analyticsView').style.display = tab === 'analytics' ? '' : 'none';
  document.getElementById('logsView').style.display = tab === 'logs' ? '' : 'none';
  document.getElementById('settingsView').style.display = tab === 'settings' ? '' : 'none';
  document.getElementById('diagnosticsView').style.display = tab === 'diagnostics' ? '' : 'none';
  document.getElementById('tabDashboardBtn').classList.toggle('active', tab === 'dashboard');
  document.getElementById('tabAnalyticsBtn').classList.toggle('active', tab === 'analytics');
  document.getElementById('tabLogsBtn').classList.toggle('active', tab === 'logs');
  document.getElementById('tabSettingsBtn').classList.toggle('active', tab === 'settings');
  document.getElementById('tabDiagnosticsBtn').classList.toggle('active', tab === 'diagnostics');
  if (tab === 'settings' && !strategySettingsLoaded) loadStrategySettings();
  if (tab === 'analytics') {
    if (!tradesFilterLoaded) loadTradesFilterOptions();
    loadTradesTable();
    loadPatterns();
    loadFeaturesCount();
  }
  if (tab === 'logs') {
    if (!logsFilterLoaded) loadLogsFilterOptions();
    renderActivityLog();
  }
  if (tab === 'diagnostics') loadDiagnostics();
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

function patternsTableHtml(headLabel, rows){
  let html = `<tr><th>${headLabel}</th><th>Сделок</th><th>W/L</th><th>Winrate</th><th>PnL</th></tr>`;
  for (const x of rows) {
    html += `<tr><td>${x.label}</td><td>${x.trades}</td><td>${x.wins}W/${x.losses}L</td>
          <td>${x.winrate_pct.toFixed(1)}%</td><td class="${cls(x.net_pnl)}">${money(x.net_pnl)}</td></tr>`;
  }
  return html;
}

async function loadPatterns(){
  const r = await fetch('/api/patterns');
  const d = await r.json();

  // Winrate as bar height (comparable, bounded 0-100%); color by PnL
  // sign (status, same green/red the rest of the page already uses) —
  // the exact trade count rides the value label so a thin, low-sample
  // bar never gets mistaken for a well-supported one.
  const hourItems = d.by_hour.map(x => ({
    label: x.label, value: x.winrate_pct, color: x.net_pnl >= 0 ? STATUS_GOOD : STATUS_BAD,
    valueLabel: `${x.winrate_pct.toFixed(0)}% (${x.trades})`,
  }));
  document.getElementById('byHourChart').innerHTML = svgBarChart(hourItems, 900, 220);

  document.getElementById('byVolatility').innerHTML = patternsTableHtml('Режим', d.by_volatility);
  document.getElementById('byTrend').innerHTML = patternsTableHtml('Тренд', d.by_trend);
}

function exportFeaturesCsv(){
  window.location.href = '/api/features/export.csv';
}

async function loadFeaturesCount(){
  const r = await fetch('/api/features/count');
  const d = await r.json();
  document.getElementById('featuresCount').textContent = d.count.toLocaleString('ru-RU');
}

// Диагностика tab — the same per-strategy "why is it (not) trading"
// breakdown scripts/decision_breakdown.py prints from a terminal (see
// its docstring for what each raw `decision` value means), live in the
// dashboard. Collapsed from up to 9 raw decision strings into 4
// semantic, STATUS-colored buckets (this is state, not series identity
// — see the dataviz skill: status tokens are reserved for state and
// never reused as "series N"): a 9-hue categorical legend would blur
// past the palette's own 8-color CVD-safe limit for no real benefit,
// since most of those 9 raw values are rare technical edge cases
// (no live quote / balance too small / slippage) that read the same
// either way — "blocked, not interesting right now" — while the one
// genuinely actionable rejection (priced out by max_coefficient) gets
// its own distinct color precisely because it's the one worth acting on.
const DIAG_COLORS = { opened: STATUS_GOOD, quiet: '#5a5d63', blocked_price: '#c98500', blocked_other: STATUS_BAD };
const DIAG_LABELS = {
  opened: 'Открыто', quiet: 'Без сигнала / уже в позиции',
  blocked_price: 'Отклонено по цене (max_coefficient)', blocked_other: 'Прочий отказ',
};

function bucketDecisions(decisions){
  const out = { opened: 0, quiet: 0, blocked_price: 0, blocked_other: 0 };
  for (const [decision, count] of Object.entries(decisions)) {
    if (decision === 'opened') out.opened += count;
    else if (decision === 'no_signal' || decision === 'skipped_already_positioned') out.quiet += count;
    else if (decision === 'rejected_max_coefficient') out.blocked_price += count;
    else out.blocked_other += count;   // rejected_no_quote/low_balance/no_fill_price/max_slippage/insufficient_funds/unknown
  }
  return out;
}

async function loadDiagnostics(){
  const hours = document.getElementById('diagPeriod').value;
  const url = '/api/diagnostics/decision_breakdown' + (hours ? `?hours=${hours}` : '');
  const r = await fetch(url);
  const d = await r.json();
  renderDiagnostics(d.strategies || []);
}

function renderDiagnostics(strategies){
  document.getElementById('diagLegend').innerHTML = Object.keys(DIAG_COLORS).map(key =>
    `<div class="legend-item"><span class="legend-dot" style="background:${DIAG_COLORS[key]}"></span>${DIAG_LABELS[key]}</div>`
  ).join('');

  if (strategies.length === 0) {
    document.getElementById('diagTable').innerHTML =
      '<tr><td>Нет оценок чекпоинта за выбранный период — попробуй «За всё время».</td></tr>';
    return;
  }

  let html = '<tr><th>Стратегия</th><th>Всего</th><th>Открыто</th><th class="diag-bar-cell">Разбивка</th>' +
             '<th class="diag-price-stats">Цены отклонённых (p50 / p75 / p90 / max)</th></tr>';
  for (const s of strategies) {
    const b = bucketDecisions(s.decisions);
    const opened = b.opened;
    let bar = '<div class="diag-stackbar">';
    for (const key of ['opened', 'quiet', 'blocked_price', 'blocked_other']) {
      const count = b[key];
      if (count === 0) continue;
      const pct = (count / s.total * 100).toFixed(1);
      bar += `<div class="seg" style="width:${pct}%;background:${DIAG_COLORS[key]}" ` +
             `title="${DIAG_LABELS[key]}: ${count} (${pct}%)"></div>`;
    }
    bar += '</div>';
    const ps = s.rejected_price_stats;
    const priceStats = ps
      ? `${ps.p50.toFixed(3)} / ${ps.p75.toFixed(3)} / ${ps.p90.toFixed(3)} / ${ps.max.toFixed(3)} (n=${ps.n})`
      : '—';
    html += `<tr><td>${escapeHtml(s.display_name)}</td><td>${s.total}</td><td>${opened}</td>` +
            `<td class="diag-bar-cell">${bar}</td><td class="diag-price-stats">${priceStats}</td></tr>`;
  }
  document.getElementById('diagTable').innerHTML = html;
}

// -- tiny inline-SVG charts (no charting library) ----------------------------------
// Mark specs kept consistent across every chart below: hairline recessive
// gridlines/axes (#2c2c2a), muted axis/label text (#898781), 2px round-cap
// lines, bars capped at 24px with a 4px rounded data-end + 2px gap between
// neighbors, value labels at the bar tip (never inside — these bars are
// often too short for an inline label to fit), a native <title> per mark
// as the hover layer (a real tooltip, just the browser's own rather than a
// custom JS one).
// Plots equity INDEXED to each series' own starting balance (idx=100 at
// t0), not raw dollars — strategies here have different deposit_usd AND
// different entry-window counts (a 3-window strategy's wallets sum to
// $300 starting, a dynamic_timing strategy's one shared wallet starts at
// $100), so a shared dollar axis let baseline size alone dominate the
// chart (one pair of $300-summed strategies stretching the axis to
// ~$319 while everyone else's real PnL swings sat invisible near the
// bottom) — exactly the dataviz skill's "two measures of different
// scale -> index to a common base (=100 at t0) on ONE axis" anti-pattern
// fix, not a dual axis. `p.idx` is precomputed by the caller as
// p.equity / series_initial_balance * 100; `p.equity` is kept on each
// point only for the tooltip, never plotted.
function svgLineChart(series, width, height){
  const pad = 34;
  let allPoints = series.flatMap(s => s.points);
  if (allPoints.length === 0) return '<svg viewBox="0 0 ' + width + ' ' + height + '"></svg>';
  const tMin = Math.min(...allPoints.map(p => p.t)), tMax = Math.max(...allPoints.map(p => p.t));
  // Baseline (100%) always inside the visible range, even if every
  // series stayed flat or all moved the same direction — otherwise the
  // one fixed reference point readers compare everything against could
  // silently fall off the plotted band.
  const idxMin = Math.min(100, ...allPoints.map(p => p.idx)), idxMax = Math.max(100, ...allPoints.map(p => p.idx));
  const idxSpan = (idxMax - idxMin) || 1, tSpan = (tMax - tMin) || 1;
  const x = t => pad + (t - tMin) / tSpan * (width - 2 * pad);
  const y = v => height - pad - (v - idxMin) / idxSpan * (height - 2 * pad);

  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<line x1="${pad}" y1="${height-pad}" x2="${width-pad}" y2="${height-pad}" stroke="#2c2c2a"/>`;
  svg += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height-pad}" stroke="#2c2c2a"/>`;
  svg += `<text x="4" y="${pad+4}" fill="#898781" font-size="10">${idxMax.toFixed(1)}%</text>`;
  svg += `<text x="4" y="${height-pad}" fill="#898781" font-size="10">${idxMin.toFixed(1)}%</text>`;
  // Baseline reference at 100% — solid hairline (never dashed, see the
  // dataviz skill's anti-patterns), one shade brighter than the axes so
  // it's readable as "start" without competing with the data lines.
  svg += `<line x1="${pad}" y1="${y(100)}" x2="${width-pad}" y2="${y(100)}" stroke="#3a3d42"/>`;
  svg += `<text x="${width-pad-2}" y="${y(100)-4}" fill="#898781" font-size="9" text-anchor="end">старт (100%)</text>`;
  series.forEach((s, i) => {
    if (s.points.length === 0) return;
    const d = s.points.map(p => `${x(p.t)},${y(p.idx)}`).join(' ');
    svg += `<polyline points="${d}" fill="none" stroke="${s.color}" stroke-width="2" ` +
           `stroke-linecap="round" stroke-linejoin="round"><title>${escapeHtml(s.name)}</title></polyline>`;
    const last = s.points[s.points.length - 1];
    svg += `<circle cx="${x(last.t)}" cy="${y(last.idx)}" r="5" fill="${s.color}" stroke="#171a21" stroke-width="2">` +
           `<title>${escapeHtml(s.name)}: ${last.idx.toFixed(1)}% ($${last.equity.toFixed(2)})</title></circle>`;
  });
  svg += '</svg>';
  return svg;
}

function svgBarChart(items, width, height){
  const pad = 34;
  if (items.length === 0) return `<svg viewBox="0 0 ${width} ${height}"></svg>`;
  const vMax = Math.max(1, ...items.map(it => Math.abs(it.value)));
  const zeroY = height - pad;
  const slot = (width - 2 * pad) / items.length;
  const barWidth = Math.min(24, slot - 4);
  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<line x1="${pad}" y1="${zeroY}" x2="${width-pad}" y2="${zeroY}" stroke="#2c2c2a"/>`;
  items.forEach((it, i) => {
    const h = Math.max(Math.abs(it.value) / vMax * (height - 2 * pad - 20), 1);
    const barX = pad + i * slot + (slot - barWidth) / 2;
    const barY = it.value >= 0 ? zeroY - h : zeroY;
    const r = Math.min(4, barWidth / 2, h);
    const valueLabel = it.valueLabel ?? it.value.toFixed(1);
    svg += `<rect x="${barX}" y="${barY}" width="${barWidth}" height="${h}" fill="${it.color}" rx="${r}" ry="${r}">` +
           `<title>${escapeHtml(it.label)}: ${valueLabel}</title></rect>`;
    svg += `<text x="${barX + barWidth/2}" y="${zeroY + 14}" fill="#898781" font-size="9" text-anchor="middle">` +
           `${it.label.length > 10 ? it.label.slice(0,10)+'…' : it.label}</text>`;
    svg += `<text x="${barX + barWidth/2}" y="${it.value >= 0 ? barY - 4 : barY + h + 12}" fill="#e6e6e6" ` +
           `font-size="9" text-anchor="middle">${valueLabel}</text>`;
  });
  svg += '</svg>';
  return svg;
}

// Horizontal bars — the right form for MANY named categories (see the
// dataviz skill's "more than ~7 classes -> a table, or table+chart":
// this reads as both at once, label + exact value + bar, no rotated or
// truncated-past-recognition labels the way a crowded vertical chart
// forces). Height grows with item count rather than cramming everything
// into a fixed box, so the card just gets taller, not denser.
function svgHBarChart(items, width, rowHeight){
  const labelW = 160, padRight = 46, padTop = 6;
  const plotW = width - labelW - padRight;
  const height = padTop * 2 + items.length * rowHeight;
  if (items.length === 0) return `<svg viewBox="0 0 ${width} ${Math.max(height,1)}"></svg>`;
  const vMax = Math.max(1e-9, ...items.map(it => Math.abs(it.value)));
  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
  const barH = Math.min(20, rowHeight - 6);
  items.forEach((it, i) => {
    const rowY = padTop + i * rowHeight;
    const barY = rowY + (rowHeight - barH) / 2;
    const w = Math.max(Math.abs(it.value) / vMax * plotW, 2);
    const barX = labelW;
    const r = Math.min(4, barH / 2, w);
    const label = it.label.length > 24 ? it.label.slice(0, 24) + '…' : it.label;
    const valueLabel = it.valueLabel ?? it.value.toFixed(1);
    // Measure first: a bar long enough to leave no room for the value
    // label past its tip must not just clip it off the edge of the SVG —
    // move the label INSIDE the bar, right-aligned near its tip, in dark
    // ink (every fill used here is a light/pastel tone, so dark ink
    // clears contrast without a per-color luminance check).
    const estLabelW = valueLabel.length * 6.5 + 6;
    const fitsOutside = (barX + w + 6 + estLabelW) <= width;
    svg += `<g class="hbar-row">`;
    svg += `<text x="${labelW - 8}" y="${rowY + rowHeight/2 + 3}" fill="#c3c2b7" font-size="11" ` +
           `text-anchor="end">${escapeHtml(label)}<title>${escapeHtml(it.label)}</title></text>`;
    svg += `<rect x="${barX}" y="${barY}" width="${w}" height="${barH}" fill="${it.color}" rx="${r}" ry="${r}">` +
           `<title>${escapeHtml(it.label)}: ${valueLabel}</title></rect>`;
    if (fitsOutside) {
      svg += `<text x="${barX + w + 6}" y="${rowY + rowHeight/2 + 3}" fill="#e6e6e6" font-size="11">${valueLabel}</text>`;
    } else {
      svg += `<text x="${barX + w - 6}" y="${rowY + rowHeight/2 + 3}" fill="#0b0b0b" font-size="11" ` +
             `text-anchor="end">${valueLabel}</text>`;
    }
    svg += `</g>`;
  });
  svg += '</svg>';
  return svg;
}

function renderAnalytics(d){
  // Equity curve compares STRATEGIES over time (identity job -> categorical
  // color) — the validated palette only clears its CVD gates through 8
  // series; past that, generating a 9th hue is worse than not having it
  // (see the dataviz skill's series-count ladder), so only the top 8
  // strategies by |PnL| get a line here. Nothing is hidden data-wise —
  // every strategy's own numbers are still in the PnL/wallets views below.
  const byAbsPnl = [...d.strategy_summary].sort((a, b) => Math.abs(b.net_pnl) - Math.abs(a.net_pnl));
  const shown = byAbsPnl.slice(0, 8);
  const series = shown.map((s, i) => ({
    name: s.display_name, color: PALETTE[i % PALETTE.length],
    // idx = % of this strategy's OWN starting balance — see svgLineChart's
    // comment for why (different strategies start at different $ totals
    // depending on how many entry-window wallets they sum).
    points: (s.equity_curve || []).map(p => ({ ...p, idx: p.equity / (s.initial_balance || 100) * 100 })),
  }));
  document.getElementById('equityChart').innerHTML = svgLineChart(series, 760, 260);
  const legend = series.map(s =>
    `<div class="legend-item"><span class="legend-dot" style="background:${s.color}"></span>${s.name}</div>`
  ).join('');
  document.getElementById('equityChart').innerHTML += `<div class="legend">${legend}</div>`;
  document.getElementById('equityNote').textContent = byAbsPnl.length > 8
    ? `Показаны 8 стратегий с наибольшим |PnL| из ${byAbsPnl.length} — остальные см. в PnL-графике ниже.`
    : '';

  // PnL/winrate are magnitude comparisons across MANY named categories —
  // horizontal bars read as a table+chart at once (label, exact value,
  // AND a bar), no rotated or truncated-past-recognition labels.
  const pnlItems = [...d.strategy_summary]
    .sort((a, b) => b.net_pnl - a.net_pnl)
    .map(s => ({ label: s.display_name, value: s.net_pnl, color: s.net_pnl >= 0 ? STATUS_GOOD : STATUS_BAD,
                 valueLabel: money(s.net_pnl) }));
  document.getElementById('pnlChart').innerHTML = svgHBarChart(pnlItems, 760, 26);

  // Winrate is a 0-100% magnitude, not a signed value -> sequential
  // single-hue color (brighter = higher), not the status green/red pair.
  const winrateItems = [...d.combos]
    .sort((a, b) => b.winrate_pct - a.winrate_pct)
    .map(c => ({
      label: `${c.display_name} @${c.window_min}м`, value: c.winrate_pct,
      color: sequentialBlue(c.winrate_pct / 100), valueLabel: `${c.winrate_pct.toFixed(1)}% (${c.trades})`,
    }));
  document.getElementById('winrateChart').innerHTML = svgHBarChart(winrateItems, 760, 24);
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

  let w = '<tr><th>Стратегия</th><th>Баланс</th><th>В сделках</th><th>Equity</th><th>PnL</th><th>W/L</th><th>Winrate</th><th>Открыто</th></tr>';
  for (const x of d.wallets) {
    w += `<tr><td>${x.display_name}</td><td>$${x.balance.toFixed(2)}</td>
          <td>$${x.reserved.toFixed(2)}</td><td>$${x.equity.toFixed(2)}</td>
          <td class="${cls(x.net_pnl)}">${money(x.net_pnl)}</td>
          <td>${x.wins}W/${x.losses}L</td><td>${x.winrate_pct.toFixed(1)}%</td><td>${x.open_trades}</td></tr>`;
  }
  document.getElementById('wallets').innerHTML = w;

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
  if (!confirm(`Сбросить только "${name}"? Все её кошельки (по каждому окну входа) и история ` +
               'сделок начнутся заново, остальные стратегии не тронет. Действие необратимо.')) return;
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

async function resetFeatures(){
  if(!confirm('Стереть ВСЮ таблицу checkpoint_features (данные для обучения)? Кошельки и история ' +
              'сделок НЕ тронутся — это отдельный лог. Действие необратимо.')) return;
  const btn = document.getElementById('resetFeaturesBtn');
  const msg = document.getElementById('resetFeaturesMsg');
  btn.disabled = true;
  msg.textContent = 'сбрасываю…';
  try {
    const r = await fetch('/api/features/reset', { method: 'POST' });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || ('HTTP ' + r.status));
    msg.textContent = `готово ✓ (удалено строк: ${j.deleted})`;
    await loadFeaturesCount();
  } catch (e) {
    msg.textContent = 'ошибка: ' + e.message;
  } finally {
    btn.disabled = false;
    setTimeout(() => { msg.textContent = ''; }, 5000);
  }
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


def _equity_curve_from_trades(trades: list, initial_balance: float, max_points: int = 300) -> list[dict]:
    """Reconstruct a (timestamp, equity) series from a set of closed
    trades — no separate snapshot table needed. Works equally for one
    wallet's own trades (see _equity_curve below) or several checkpoint
    wallets' trades merged together (see /api/state's strategy_summary,
    which needs one combined curve per STRATEGY, not one line per
    checkpoint). Downsampled to `max_points` (keeping the very last
    point) so a long-running strategy doesn't bloat the /api/state
    payload every poll."""
    closed = sorted((t for t in trades if t.closed_ts is not None), key=lambda t: t.closed_ts)
    if not closed:
        return [{"t": time.time(), "equity": initial_balance}]

    points = [{"t": closed[0].opened_ts, "equity": initial_balance}]
    equity = initial_balance
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


def _equity_curve(wallet, max_points: int = 300) -> list[dict]:
    return _equity_curve_from_trades(wallet.trades, wallet.initial_balance, max_points)


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

        # snap.wallets is keyed by Engine's composite "strategy:window_min"
        # wallet id (one wallet per entry checkpoint now, not per
        # strategy) — the base strategy name/display_name lookup has to
        # come from w.strategy, not the dict key itself. w.window_min is
        # None for a dynamic_timing strategy's one shared wallet (see
        # StrategyConfig.dynamic_timing) — labeled "(авто)" rather than
        # "(Noneм)".
        def _wallet_record(w):
            closed = w.closed_trades()
            wins = sum(1 for t in closed if t.status == TradeStatus.WON)
            losses = len(closed) - wins
            winrate_pct = (wins / len(closed) * 100) if closed else 0.0
            return {
                "strategy": w.strategy,
                "window_min": w.window_min,
                "display_name": (
                    f"{display_names.get(w.strategy, w.strategy)} "
                    + (f"({w.window_min}м)" if w.window_min is not None else "(авто)")
                ),
                "initial_balance": w.initial_balance,
                "balance": w.balance,
                "reserved": w.reserved,
                "equity": w.equity,
                "net_pnl": w.net_pnl,
                "wins": wins,
                "losses": losses,
                "winrate_pct": winrate_pct,
                "open_trades": len(w.open_trades()),
                "equity_curve": _equity_curve(w),
            }

        wallets = [
            _wallet_record(w)
            for w in sorted(snap.wallets.values(), key=lambda w: (w.strategy, -(w.window_min or 0)))
        ]

        # One combined line per STRATEGY (not per checkpoint wallet) —
        # the Analytics tab's equity/PnL charts compare strategies against
        # each other, and 20+ checkpoint-level lines/bars is unreadable
        # (see the dashboard's own history: that's exactly what this
        # replaced). Each checkpoint's own numbers are still fully visible
        # in `wallets` above and the "Стратегия × время входа" combos below.
        strategy_groups: dict[str, list] = {}
        for w in snap.wallets.values():
            strategy_groups.setdefault(w.strategy, []).append(w)
        strategy_summary = []
        for strategy_name, group in strategy_groups.items():
            initial_balance = sum(w.initial_balance for w in group)
            equity = sum(w.equity for w in group)
            all_trades = [t for w in group for t in w.trades]
            strategy_summary.append({
                "strategy": strategy_name,
                "display_name": display_names.get(strategy_name, strategy_name),
                "initial_balance": initial_balance,
                "equity": equity,
                "net_pnl": equity - initial_balance,
                "equity_curve": _equity_curve_from_trades(all_trades, initial_balance),
            })
        strategy_summary.sort(key=lambda s: s["strategy"])

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
                "strategy_summary": strategy_summary,
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

    @app.get("/api/patterns")
    async def get_patterns(strategy: Optional[str] = None) -> JSONResponse:
        """Bot-wide (or, with `strategy`, one strategy's) win/loss/PnL
        broken down by hour-of-day, realized-volatility regime, and BTC
        trend direction at entry — the Analytics tab's "Закономерности"
        cards. Reads the durable trades/checkpoint_features tables
        directly (see Storage.get_hourly_stats/get_volatility_regime_stats
        /get_trend_direction_stats), so these cover the bot's FULL
        history, not just what's still in memory since the last restart."""
        return JSONResponse(
            {
                "by_hour": engine.storage.get_hourly_stats(strategy=strategy),
                "by_volatility": engine.storage.get_volatility_regime_stats(strategy=strategy),
                "by_trend": engine.storage.get_trend_direction_stats(strategy=strategy),
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

    @app.get("/api/features/export.csv")
    async def export_features_csv(strategy: Optional[str] = None) -> StreamingResponse:
        """Full checkpoint_features history (every checkpoint EVALUATION,
        no_signal/rejected/opened alike — see storage.py's module
        docstring for why the negative examples matter for training too)
        as a downloadable CSV. Unfiltered by default — this is meant to
        leave with the whole dataset, not one page of it."""
        rows = engine.storage.get_checkpoint_features(strategy=strategy, limit=None)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        filename = f"checkpoint_features_{strategy or 'all'}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            iter([buf.getvalue()]), media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/features/count")
    async def features_count(strategy: Optional[str] = None) -> JSONResponse:
        return JSONResponse({"count": engine.storage.count_checkpoint_features(strategy=strategy)})

    @app.post("/api/features/reset")
    async def reset_features() -> JSONResponse:
        """Wipe checkpoint_features specifically — the one table the
        regular Reset DB button deliberately leaves alone (see
        Storage.reset's docstring). Separate, explicit action for when
        the accumulated ML history itself is what needs to go (e.g. after
        dropping 5MIN contracts made old rows no longer representative of
        what's currently traded) — never a side effect of a balance
        reset. Wallets/trades are untouched."""
        deleted = engine.storage.reset_checkpoint_features()
        return JSONResponse({"ok": True, "deleted": deleted, "reset_at": time.time()})

    @app.get("/api/diagnostics/decision_breakdown")
    async def diagnostics_decision_breakdown(hours: Optional[float] = None) -> JSONResponse:
        """Backs the Диагностика tab — the same per-strategy 'why is it
        (not) trading' breakdown scripts/decision_breakdown.py prints
        from a terminal, live in the dashboard instead. See that
        script's docstring for what each decision value means.

        `hours` (omit for all-time): checkpoint_features is never wiped
        by a Reset, so all-time counts silently mix evaluations from
        before the strategy's wallet was last reset (or its logic last
        changed) with its actual current trade history — see
        Storage.get_decision_breakdown's docstring for the live case
        that motivated this filter."""
        since_ts = (time.time() - hours * 3600) if hours else None
        rows = engine.storage.get_decision_breakdown(since_ts=since_ts)
        for row in rows:
            row["display_name"] = display_names.get(row["strategy"], row["strategy"])
            row["rejected_price_stats"] = engine.storage.get_rejected_fill_price_stats(
                row["strategy"], since_ts=since_ts,
            )
        return JSONResponse({"strategies": rows})

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
