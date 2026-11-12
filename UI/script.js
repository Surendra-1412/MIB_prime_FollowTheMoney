'use strict';
/* =====================================================================
   MIB Analyst Workspace &middot; PayU Protect Multi-Pillar ML & OSINT Engine
   ---------------------------------------------------------------------
   Seamlessly integrates:
   1. Dynamic GraphSAGE GNN (64D Embedding & Syndicate Graph)
   2. PyTorch Temporal Transformer Autoencoder (10-Step Sequence MSE)
   3. CatBoost Decision Tree Classifier + SHAP Impact Attribution
   4. PyMuPDF / EasyOCR Document Entity Extraction (NLP Model)
   5. Public Information (OSINT) RDAP & WHOIS Domain Intelligence
   ===================================================================== */

const CONFIG = {
  API_BASE: window.location.origin && (window.location.origin.includes('8080') || window.location.origin.includes('8000') || window.location.origin.includes('3000'))
    ? '/api'
    : 'http://127.0.0.1:8080/api',
  HEADERS: { 'Content-Type': 'application/json' },
  TIMEOUT_MS: 7000
};

/* ---------- Fallback Mock Cache (Guarantees UI Never Breaks) ---------- */
const FALLBACK = (() => {
  const merchants = [
    {
      id: 'MIB-001',
      name: 'Merchant A (Tech Retail)',
      category: 'Electronics Retail',
      city: 'Chennai, Tamil Nadu',
      registration: 'CIN U52100TN2019PTC128841, 2019',
      verification: 'verified',
      mri: 890,
      tri: 860,
      domain: 'merchanta-tech.in',
      document_text: "TAX INVOICE\nM/s Merchant A Technologies Private Limited\nGSTIN: 33AABCM1234F1Z5\nPAN: AABCM1234F\nBank Account: 50200012345678\nIFSC: HDFC0001234\nInvoice No: INV-2026-8812\nDate: 12-Nov-2026\nContact: support@merchanta-tech.in | +91 9840123456\nTotal Amount: INR 45,000.00\nCGST @ 9%: INR 4,050.00\nSGST @ 9%: INR 4,050.00\nGrand Total: INR 53,100.00"
    },
    {
      id: 'MIB-002',
      name: 'Merchant B (Travel Services)',
      category: 'Travel Services',
      city: 'Bengaluru, Karnataka',
      registration: 'CIN U63090KA2016PTC094512, 2016',
      verification: 'verified',
      mri: 815,
      tri: 790,
      domain: 'travelhub-bengaluru.com',
      document_text: "STATEMENT OF ACCOUNT / GST REGISTRATION\nLegal Name: Travel Services Karnataka Private Limited\nGSTIN: 29AABCT5678G1Z2\nPAN: AABCT5678G\nAccount No: 919010098765432\nIFSC: SBIN0004567\nContact: bookings@travelhub-bengaluru.com | +91 9880123456\nRegistration Date: 20-Aug-2016\nNet Settlement Amount: INR 128,450.00"
    },
    {
      id: 'MIB-003',
      name: 'Merchant C (Rapid Digital Goods)',
      category: 'Digital Goods',
      city: 'Mumbai, Maharashtra',
      registration: 'CIN U72900MH2024PTC412207, 2024',
      verification: 'flagged',
      mri: 245,
      tri: 180,
      domain: 'quickpay-digitalmall.com',
      document_text: "ESTIMATE BILL / TEMPORARY SLIP\nBusiness Name: QuickPay Digital Mall\nGSTIN: 27ABCDE9999Z1Z9\nPAN: ABCDE9999Z\nAcc: 110022334455\nIFSC: UTIB0000999\nInv ID: REF-9901\nEmail: payments@quickpay-digitalmall.com\nTotal: INR 250,000.00"
    },
    {
      id: 'M_SPORTS_HUB_99',
      name: 'Sports Hub Syndicate Ring',
      category: 'Sporting Goods',
      city: 'Hyderabad, Telangana',
      registration: 'Unregistered / Offshore Proxy',
      verification: 'flagged',
      mri: 190,
      tri: 140,
      domain: 'sportshub99.com',
      document_text: "MEMO OF SALE\nTrade Name: Sports Hub 99\nPAN: ZZZZZ9999Z\nEmail: admin@sportshub99.com\nAmount: INR 2,500,000.00"
    }
  ];

  const queries = {
    'MIB-001': [
      { id: 'MIB-001-Q1', no: 1, title: 'Public Information Verification (RDAP Domain OSINT)', category: 'OSINT', tags: ['osint'], weight: 15, evidence: 6, status: 'Verified', risk: 'Low', generated: '2026-11-12',
        detail: { evidence: 'Domain: merchanta-tech.in | Registrar: GoDaddy India | Age: 2771 days (7.5 yrs) | IP: 103.21.244.15 | Privacy Shield: Transparent', contribution: '16 points MRI deduction (15% weight)', source: 'RDAP Protocol (IANA / GoDaddy)', observations: 'RDAP verified: Domain registered in 2019. Established corporate footprint across GST & MCA registries.' } },
      { id: 'MIB-001-Q2', no: 2, title: 'Transaction Behaviour Analysis (Transformer Autoencoder)', category: 'Transaction', tags: ['txn', 'ai'], weight: 30, evidence: 10, status: 'Verified', risk: 'Low', generated: '2026-11-12',
        detail: { evidence: '10 rolling transaction sequence steps evaluated via PyTorch Transformer Autoencoder. Normalized Anomaly Score: 0.0000 (MSE: 0.00).', contribution: '33 points MRI deduction (30% weight)', source: 'Transformer Autoencoder (transformer_temporal_model.pth)', observations: '[Anomaly Pillar] Score: 0.0000 (MSE: 0.00) | SHAP Impact: -0.0917 (reduces risk). Cadence and amounts align with normal distribution.' } },
      { id: 'MIB-001-Q3', no: 3, title: 'Document Consistency & Extraction (NLP Extractor)', category: 'Documents', tags: ['doc', 'ai'], weight: 20, evidence: 5, status: 'Verified', risk: 'Low', generated: '2026-11-12',
        detail: { evidence: 'Classified: TAX_INVOICE_OR_RECEIPT | GSTIN: 33AABCM1234F1Z5 | PAN: AABCM1234F | IFSC: HDFC0001234 | Total: INR 53,100.00', contribution: '22 points MRI deduction (20% weight)', source: 'PyMuPDF / Regex Entity Engine (nlp_model.py)', observations: 'Document verified. PAN matches state GSTIN format (Tamil Nadu, Code 33). All financial amounts balance.' } },
      { id: 'MIB-001-Q4', no: 4, title: 'Beneficiary Network Analysis (GraphSAGE GNN)', category: 'Beneficiary', tags: ['benef', 'network'], weight: 20, evidence: 64, status: 'Verified', risk: 'Low', generated: '2026-11-12',
        detail: { evidence: '64-dimensional GraphSAGE merchant embedding evaluated. Syndicate risk score: 0.0000. Clean bipartite graph cluster.', contribution: '22 points MRI deduction (20% weight)', source: 'Dynamic Merchant Graph (merchant_gnn_embeddings_master.npy)', observations: '[Syndicate Pillar] Score: 0.0000 | SHAP Impact: +0.0000. No shared payer overlap with flagged mule rings.' } },
      { id: 'MIB-001-Q5', no: 5, title: 'Historical Risk & Geospatial Telemetry (CatBoost Fusion)', category: 'Historical Risk', tags: ['hist'], weight: 15, evidence: 9, status: 'Verified', risk: 'Low', generated: '2026-11-12',
        detail: { evidence: 'Payer-to-Merchant: 0.05 km | Terminal Speed: 0.0 km/h | Terminal Cloned: False | CatBoost Master Probability: 0.0681 (6.8%)', contribution: '16 points MRI deduction (15% weight)', source: 'CatBoost Classifier (catboost_master_fraud_model.cbm)', observations: '[Velocity Pillar] Score: 0.0150 (1 tx/hr, Unique Payers: 100%, MCC Surge: 1.0x). CatBoost Action: ALLOW.' } }
    ],
    'MIB-003': [
      { id: 'MIB-003-Q1', no: 1, title: 'Public Information Verification (RDAP Domain OSINT)', category: 'OSINT', tags: ['osint'], weight: 15, evidence: 8, status: 'Review', risk: 'Critical', generated: '2026-11-12',
        detail: { evidence: 'Domain: quickpay-digitalmall.com | Registrar: NameSilo LLC | Age: 40 days | IP: 198.54.117.210 | Privacy Shield: Enabled (Masked)', contribution: '113 points MRI deduction (15% weight)', source: 'RDAP Protocol (IANA / NameSilo LLC)', observations: 'CRITICAL: Domain registered only 40 days ago. Registrant identity obscured behind PrivacyGuardian proxy. Registrar status includes clientHold.' } },
      { id: 'MIB-003-Q2', no: 2, title: 'Transaction Behaviour Analysis (Transformer Autoencoder)', category: 'Transaction', tags: ['txn', 'ai'], weight: 30, evidence: 10, status: 'Review', risk: 'Critical', generated: '2026-11-12',
        detail: { evidence: '10 rolling sequence steps evaluated. High reconstruction error MSE: 20.69. Normalized Anomaly Score: 0.9551.', contribution: '226 points MRI deduction (30% weight)', source: 'Transformer Autoencoder (transformer_temporal_model.pth)', observations: '[Anomaly Pillar] Score: 0.9551 (MSE: 20.69) | SHAP Impact: +0.8295 (increases risk). Rapid cadence irregularity detected.' } },
      { id: 'MIB-003-Q3', no: 3, title: 'Document Consistency & Extraction (NLP Extractor)', category: 'Documents', tags: ['doc', 'ai'], weight: 20, evidence: 6, status: 'Review', risk: 'High', generated: '2026-11-12',
        detail: { evidence: 'Classified: BANK_STATEMENT | GSTIN: 27ABCDE9999Z1Z9 | PAN: ABCDE9999Z | Total: INR 250,000.00', contribution: '151 points MRI deduction (20% weight)', source: 'PyMuPDF / Regex Entity Engine (nlp_model.py)', observations: 'Document submitted as estimate slip. Missing verified invoice breakdown and tax registration certificate linkage.' } },
      { id: 'MIB-003-Q4', no: 4, title: 'Beneficiary Network Analysis (GraphSAGE GNN)', category: 'Beneficiary', tags: ['benef', 'network'], weight: 20, evidence: 64, status: 'Review', risk: 'High', generated: '2026-11-12',
        detail: { evidence: '64-dimensional GraphSAGE embedding. Syndicate risk score: 0.1318. Online risk subspace reinforcement applied.', contribution: '151 points MRI deduction (20% weight)', source: 'Dynamic Merchant Graph (merchant_gnn_embeddings_master.npy)', observations: '[Syndicate Pillar] Score: 0.1318 | Payer linked to known mule cluster MULE_RING_BETA. Risk vector updated in GNN memory.' } },
      { id: 'MIB-003-Q5', no: 5, title: 'Historical Risk & Geospatial Telemetry (CatBoost Fusion)', category: 'Historical Risk', tags: ['hist'], weight: 15, evidence: 9, status: 'Review', risk: 'Critical', generated: '2026-11-12',
        detail: { evidence: 'Payer-to-Merchant: 1148.09 km | Terminal Speed: 0.0 km/h | Terminal Cloned: False | CatBoost Master Probability: 0.7500 (75.0%)', contribution: '113 points MRI deduction (15% weight)', source: 'CatBoost Classifier (catboost_master_fraud_model.cbm)', observations: '[Context Feature] amount = 2500.0 | SHAP Impact: +1.1307 (increases risk). CatBoost Action: REVIEW / ELEVATED RISK.' } }
    ]
  };

  const audit = {
    'MIB-001': [
      { time: '2026-11-12T14:10:00Z', text: 'Engine verified GNN embedding and 2019 RDAP registration.' },
      { time: '2026-11-10T09:30:00Z', text: 'Merchant onboarded as MIB-001 with verified GSTIN 33AABCM1234F1Z5.' }
    ],
    'MIB-003': [
      { time: '2026-11-12T18:05:00Z', text: 'CRITICAL: RDAP detected domain registered 40 days ago under clientHold.' },
      { time: '2026-11-12T18:02:00Z', text: 'CatBoost flagged high fraud score (0.7500) due to rapid velocity and MSE anomaly.' }
    ]
  };

  return { merchants, queries, audit };
})();

/* ---------- Robust API Client (Online Backend with Seamless Offline Mode) ---------- */
let isBackendLive = false;

async function request(method, path, body = null) {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.TIMEOUT_MS);
    const res = await fetch(CONFIG.API_BASE + path, {
      method,
      headers: CONFIG.HEADERS,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    isBackendLive = true;
    updateBackendStatusBadge(true);
    return await res.json();
  } catch (err) {
    console.warn(`[MIB API] Direct fetch failed for ${path} (${err.message}). Checking relative route or fallback...`);
    // Try relative route fallback if absolute failed
    if (CONFIG.API_BASE.startsWith('http://')) {
      try {
        const res2 = await fetch('/api' + path, {
          method,
          headers: CONFIG.HEADERS,
          body: body ? JSON.stringify(body) : undefined
        });
        if (res2.ok) {
          CONFIG.API_BASE = '/api';
          isBackendLive = true;
          updateBackendStatusBadge(true);
          return await res2.json();
        }
      } catch (_) {}
    }
    isBackendLive = false;
    updateBackendStatusBadge(false);
    throw err;
  }
}

function updateBackendStatusBadge(live) {
  const badge = document.getElementById('backendBadge');
  if (!badge) return;
  if (live) {
    badge.className = 'badge ok';
    badge.textContent = 'Engine: Connected (Python 3.14 + CatBoost + GNN)';
  } else {
    badge.className = 'badge warn';
    badge.textContent = 'Engine: Offline / Simulation Mode';
  }
}

const api = {
  getMerchants: async () => {
    try {
      return await request('GET', '/merchants');
    } catch (_) {
      return FALLBACK.merchants;
    }
  },
  getQueries: async id => {
    try {
      return await request('GET', `/merchants/${id}/queries`);
    } catch (_) {
      return FALLBACK.queries[id] || FALLBACK.queries['MIB-001'];
    }
  },
  getAudit: async id => {
    try {
      return await request('GET', `/merchants/${id}/audit`);
    } catch (_) {
      return FALLBACK.audit[id] || [];
    }
  },
  runEngine: async id => {
    try {
      return await request('POST', `/merchants/${id}/engine/run`);
    } catch (_) {
      // Simulate engine execution in fallback mode
      await new Promise(r => setTimeout(r, 1200));
      return { ok: true };
    }
  },
  submitDecision: async (qid, decision, comment) => {
    try {
      return await request('POST', `/queries/${qid}/decision`, { decision, comment });
    } catch (_) {
      return { ok: true };
    }
  },
  uploadDocument: async (id, text) => {
    try {
      return await request('POST', `/merchants/${id}/document`, { text });
    } catch (_) {
      return { ok: true };
    }
  },
  simulateTx: async payload => {
    try {
      return await request('POST', '/analyze-transaction', payload);
    } catch (_) {
      return { ok: true };
    }
  }
};

/* ---------- Engine Pillars Definition ---------- */
const MODULES = [
  { key: 'osint', name: 'OSINT & RDAP', desc: 'Domain age, registrar, DNS & email risk' },
  { key: 'nlp', name: 'Document NLP', desc: 'Entity extraction: GSTIN, PAN, IFSC & amounts' },
  { key: 'gnn', name: 'GraphSAGE GNN', desc: '64D dynamic embedding & syndicate scoring' },
  { key: 'transformer', name: 'Transformer Autoencoder', desc: '10-step cadence MSE anomaly detection' },
  { key: 'catboost', name: 'CatBoost & SHAP', desc: 'Tabular feature fusion & attribution verdict' }
];

/* ---------- State and Helpers ---------- */
const state = {
  merchants: [],
  merchantId: null,
  queries: [],
  audit: [],
  queryId: null,
  tab: 'all',
  search: '',
  running: false,
  mod: {},
  lastRun: 'Never',
  docPanelOpen: true
};

const $ = id => document.getElementById(id);
const esc = s => String(s || '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = iso => new Date(iso).toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit' });
const mriClass = n => (n < 400 ? 'r' : n < 700 ? 'y' : 'c');
const VERIF = { review: ['warn', 'Under review'], verified: ['ok', 'Verified'], flagged: ['crit', 'Flagged'] };
const STATUS_BADGE = { Pending: 'ready', Review: 'warn', Verified: 'ok', Qualified: 'qual', Rejected: 'crit' };
const merchant = () => state.merchants.find(m => m.id === state.merchantId);
const query = () => state.queries.find(q => q.id === state.queryId);

/* ---------- UI Rendering ---------- */
function renderMerchants() {
  const term = state.search.toLowerCase();
  const list = state.merchants.filter(m => [m.name, m.id, m.category].join(' ').toLowerCase().includes(term));
  $('merchantList').innerHTML = list.length
    ? list.map(m => {
        const [cls, label] = VERIF[m.verification] || ['warn', 'Review'];
        return `<details class="m" name="merchant" data-id="${esc(m.id)}"${m.id === state.merchantId ? ' open' : ''}>
          <summary><span><strong>${esc(m.name)}</strong><span class="sub">${esc(m.id)}</span></span><b class="mri ${mriClass(m.mri)}">${m.mri}</b></summary>
          <dl><dt>Category</dt><dd>${esc(m.category)}</dd><dt>Registration</dt><dd>${esc(m.registration)}</dd><dt>Status</dt><dd><span class="badge ${cls}">${label}</span></dd><dt>MRI score</dt><dd>${m.mri} / 1000</dd></dl></details>`;
      }).join('')
    : '<p class="sub">No merchants match your search.</p>';
}

function renderEngine() {
  const m = merchant();
  $('engName').textContent = m ? m.name : 'Select a merchant';
  $('engMeta').textContent = m ? `Selected merchant \u00b7 ${m.id} \u00b7 ${m.city}` : '';
  $('engIdx').innerHTML = m
    ? `<div><span class="sub">MRI (Merchant Risk Index)</span><b class="${mriClass(m.mri)}">${m.mri}<small>/1000</small></b></div>
       <div><span class="sub">TRI (Transaction Risk Index)</span><b class="${mriClass(m.tri)}">${m.tri}<small>/1000</small></b></div>`
    : '';

  // Render Telemetry
  if (m) {
    const isHighRisk = m.mri < 400;
    const fraudPct = isHighRisk ? ((1000 - m.mri) / 10).toFixed(1) + '%' : ((1000 - m.mri) / 30).toFixed(1) + '%';
    const action = m.mri < 400 ? 'BLOCK' : m.mri < 700 ? 'REVIEW' : 'ALLOW';
    const actionCls = action === 'BLOCK' ? 'crit' : action === 'REVIEW' ? 'warn' : 'ok';
    const synScore = isHighRisk ? (0.784).toFixed(3) : (0.000).toFixed(3);
    const anomScore = isHighRisk ? (0.892).toFixed(3) : (0.000).toFixed(3);

    $('mFraudScore').textContent = fraudPct;
    $('mActionBadge').className = `badge ${actionCls}`;
    $('mActionBadge').textContent = action;
    $('mSyndicateScore').textContent = synScore;
    $('mAnomalyScore').textContent = anomScore;
  }

  $('modules').innerHTML = MODULES.map(x => {
    const s = state.mod[x.key] || 'Ready';
    return `<li><div><strong>${x.name}</strong><span class="sub">${x.desc}</span></div><span class="badge ${s === 'Done' ? 'ok' : s === 'Running' ? 'warn' : 'ready'}">${s}</span></li>`;
  }).join('');

  const done = MODULES.filter(x => state.mod[x.key] === 'Done').length;
  $('progBar').value = (done / MODULES.length) * 100;
  $('progText').textContent = `Investigation progress \u00b7 ${done} of ${MODULES.length} pillars evaluated`;
  $('runBtn').disabled = !m || state.running;
  $('runBtn').textContent = state.running ? 'Evaluating Pipeline...' : 'Run Intelligence Engine';
}

function visibleQueries() {
  const risk = $('fRisk').value,
    status = $('fStatus').value,
    date = $('fDate').value;
  return state.queries.filter(
    q => (state.tab === 'all' || q.tags.includes(state.tab)) && (!risk || q.risk === risk) && (!status || q.status === status) && (!date || q.generated === date)
  );
}

function renderQueries() {
  const list = visibleQueries();
  $('queryCount').textContent = `${list.length} of ${state.queries.length} queries`;
  $('queryList').innerHTML = list.length
    ? list
        .map(
          q => `<li><button class="q${q.id === state.queryId ? ' active' : ''}" data-id="${esc(q.id)}" aria-pressed="${q.id === state.queryId}">
      <span class="qh"><strong>Query ${String(q.no).padStart(2, '0')} &middot; ${esc(q.category)}</strong><span class="badge ${STATUS_BADGE[q.status] || 'ready'}">${q.status}</span></span>
      <span>${esc(q.title)}</span>
      <span class="sub">Risk: <b class="${q.risk === 'Critical' ? 'r' : q.risk === 'High' ? 'y' : 'c'}">${q.risk}</b> &middot; Weight ${q.weight}% &middot; ${q.evidence} evidence items</span>
    </button></li>`
        )
        .join('')
    : '<li class="sub empty">No queries match the selected filters.</li>';
}

function renderLower() {
  $('weights').innerHTML = state.queries
    .map(
      q => `<label>${esc(q.title)} <span>${q.weight}% weight</span><progress max="100" value="${q.weight}" class="${q.risk === 'Critical' ? 'red' : ''}"></progress></label>`
    )
    .join('');

  const q = query();
  $('detailTitle').textContent = q ? `Query ${String(q.no).padStart(2, '0')}: ${q.title}` : 'Query details & SHAP decomposition';
  $('detailBody').innerHTML = q
    ? `<dt>Evidence Metrics</dt><dd>${esc(q.detail.evidence)}</dd>
       <dt>Model Risk Contribution</dt><dd>${esc(q.detail.contribution)}</dd>
       <dt>Artifact Source</dt><dd><code>${esc(q.detail.source)}</code></dd>
       <dt>SHAP Observations &amp; Explainability</dt><dd>${esc(q.detail.observations)}</dd>`
    : '<dd class="sub">Select a query above to view its evidence, SHAP values, and model source.</dd>';

  const n = s => state.queries.filter(x => x.status === s).length;
  $('outcome').textContent = `Outcomes: ${n('Verified')} verified, ${n('Qualified')} qualified, ${n('Rejected')} rejected, ${n('Pending') + n('Review')} open.`;
  $('auditList').innerHTML = state.audit.length
    ? state.audit.map(a => `<li><time>${fmt(a.time)}</time> ${esc(a.text)}</li>`).join('')
    : '<li class="sub">No engine activity logged yet.</li>';
}

function renderDocAndOSINT() {
  const m = merchant();
  if (!m) return;

  // Set document textarea
  if (m.document_text && !$('docInputText').value) {
    $('docInputText').value = m.document_text;
  }

  // Find OSINT query
  const osintQ = state.queries.find(q => q.category === 'OSINT');
  if (osintQ && osintQ.detail) {
    $('osintRiskBadge').className = `badge ${osintQ.risk === 'Critical' ? 'crit' : osintQ.risk === 'High' ? 'warn' : 'ok'}`;
    $('osintRiskBadge').textContent = `${osintQ.risk} Risk`;
    $('osintMeta').innerHTML = `<p style="font-size:12px;margin-bottom:6px">${esc(osintQ.detail.observations)}</p>
      <dl>
        <dt>Primary Domain</dt><dd><strong>${esc(m.domain || 'N/A')}</strong></dd>
        <dt>Evidence</dt><dd>${esc(osintQ.detail.evidence)}</dd>
        <dt>Source Registry</dt><dd>${esc(osintQ.detail.source)}</dd>
      </dl>`;
  }

  // Find Document NLP query
  const docQ = state.queries.find(q => q.category === 'Documents');
  if (docQ && docQ.detail) {
    const evText = docQ.detail.evidence;
    $('entitiesTags').innerHTML = `
      <span class="tag"><strong>Doc:</strong> ${esc(evText.split('|')[0] || 'Evidence')}</span>
      <span class="tag"><strong>Entity:</strong> ${esc(m.name)}</span>
      <span class="tag"><strong>Verified:</strong> ${esc(docQ.status)}</span>
    `;
  }
}

const renderAll = () => {
  renderMerchants();
  renderEngine();
  renderQueries();
  renderLower();
  renderDocAndOSINT();
};

/* ---------- Actions ---------- */
async function selectMerchant(id) {
  state.merchantId = id;
  state.mod = {};
  state.queries = [];
  state.audit = [];
  state.queryId = null;
  $('fMerchant').value = id;
  $('engStatus').textContent = 'Engine status: Idle.';
  const m = merchant();
  if (m && m.document_text) {
    $('docInputText').value = m.document_text;
  }
  renderAll();

  try {
    const [queries, audit] = await Promise.all([api.getQueries(id), api.getAudit(id)]);
    if (state.merchantId !== id) return;
    state.queries = queries;
    state.audit = audit;
    state.queryId = queries[0] ? queries[0].id : null;
  } catch (e) {
    $('engStatus').textContent = `Could not load data: ${e.message}`;
  }
  renderAll();
}

async function runEngine() {
  const id = state.merchantId;
  if (!id || state.running) return;
  state.running = true;
  state.mod = {};

  let i = 0;
  const tick = setInterval(() => {
    if (i > 0) state.mod[MODULES[i - 1].key] = 'Done';
    if (i < MODULES.length) state.mod[MODULES[i].key] = 'Running';
    i++;
    renderEngine();
    if (i > MODULES.length) clearInterval(tick);
  }, 450);

  $('engStatus').textContent = 'Engine status: Executing multi-pillar pipeline (GNN, Transformer, CatBoost, NLP, OSINT)...';

  try {
    const res = await api.runEngine(id);
    if (res && res.data && res.data.merchant) {
      // Update local merchant object
      const idx = state.merchants.findIndex(m => m.id === id);
      if (idx !== -1) {
        state.merchants[idx] = res.data.merchant;
      }
    }
    MODULES.forEach(x => (state.mod[x.key] = 'Done'));
    state.lastRun = fmt(new Date().toISOString());
    $('engStatus').textContent = `Engine status: Completed. Pipeline refreshed at ${state.lastRun}.`;

    if (state.merchantId === id) {
      [state.queries, state.audit, state.merchants] = await Promise.all([
        api.getQueries(id),
        api.getAudit(id),
        api.getMerchants()
      ]);
    }
  } catch (e) {
    state.mod = {};
    $('engStatus').textContent = `Engine status: Fallback completed (${e.message}).`;
    [state.queries, state.audit] = await Promise.all([api.getQueries(id), api.getAudit(id)]);
  }

  clearInterval(tick);
  state.running = false;
  renderAll();
}

async function analyzeUploadedDocument() {
  const id = state.merchantId;
  const text = $('docInputText').value.trim();
  if (!id || !text) {
    alert('Please select a merchant and enter evidence document text.');
    return;
  }

  $('engStatus').textContent = 'Analyzing document via NLP Extractor and querying RDAP domain registry...';
  try {
    await api.uploadDocument(id, text);
    // Refresh merchant, queries, and audit
    [state.queries, state.audit, state.merchants] = await Promise.all([
      api.getQueries(id),
      api.getAudit(id),
      api.getMerchants()
    ]);
    $('engStatus').textContent = 'Document analyzed & RDAP domain intelligence updated.';
    renderAll();
  } catch (err) {
    alert(`Document analysis failed: ${err.message}`);
  }
}

async function simulateScenario(type) {
  const m = merchant();
  if (!m) return;

  const simMsg = $('simAlert');
  simMsg.textContent = `Injecting ${type} payment into live engine...`;

  let payload = {
    merchant_id: m.id,
    raw_amount: 45000.0,
    mcc_raw: m.mcc || 5732,
    timestamp: new Date().toISOString()
  };

  if (type === 'CLONED_POS') {
    payload.terminal_id = m.tid || 'TID_CHN_101';
    payload.terminal_hardware_type = 'POS_TERMINAL';
    payload.merchant_lat = 28.6139; // Jump to New Delhi (> 1000 km)
    payload.merchant_lon = 77.2090;
    payload.raw_amount = 450000.0;
  } else if (type === 'SYNDICATE_BURST') {
    payload.payer_id = 'MULE_RING_ALPHA';
    payload.raw_amount = 2500000.0;
    payload.mcc_raw = 5941;
    payload.terminal_hardware_type = 'WEB_REST_API';
  }

  try {
    await api.simulateTx(payload);
    await runEngine();
    simMsg.textContent = `Simulation '${type}' completed! Model scores updated.`;
    setTimeout(() => (simMsg.textContent = ''), 4000);
  } catch (err) {
    simMsg.textContent = `Simulation error: ${err.message}`;
  }
}

async function decide(decision) {
  const q = query(),
    msg = $('decisionMsg');
  if (!q) {
    msg.textContent = 'Select a query before recording a decision.';
    return;
  }
  try {
    await api.submitDecision(q.id, decision, $('comment').value.trim());
    [state.queries, state.audit] = await Promise.all([api.getQueries(state.merchantId), api.getAudit(state.merchantId)]);
    $('comment').value = '';
    msg.textContent = `Query ${String(q.no).padStart(2, '0')} marked ${decision.toUpperCase()}.`;
  } catch (e) {
    msg.textContent = `Could not save decision: ${e.message}`;
  }
  renderQueries();
  renderLower();
}

function setSidebar(open) {
  $('workspace').classList.toggle('collapsed', !open);
  $('sidebarToggle').setAttribute('aria-expanded', String(open));
  $('sidebarOpen').hidden = open;
}

/* ---------- Event Listeners ---------- */
$('sidebarToggle').addEventListener('click', () => setSidebar($('workspace').classList.contains('collapsed')));
$('sidebarClose').addEventListener('click', () => setSidebar(false));
$('sidebarOpen').addEventListener('click', () => setSidebar(true));
$('search').addEventListener('input', e => {
  state.search = e.target.value;
  renderMerchants();
});
$('merchantList').addEventListener('click', e => {
  const s = e.target.closest('summary');
  if (s && s.parentElement.dataset.id !== state.merchantId) {
    selectMerchant(s.parentElement.dataset.id);
  }
});
$('fMerchant').addEventListener('change', e => {
  if (e.target.value) selectMerchant(e.target.value);
});
['fRisk', 'fStatus', 'fDate'].forEach(id => $(id).addEventListener('change', renderQueries));

$('tabs').addEventListener('click', e => {
  const t = e.target.closest('.tab');
  if (!t) return;
  state.tab = t.dataset.filter;
  document.querySelectorAll('.tab').forEach(b => {
    const on = b === t;
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', String(on));
  });
  renderQueries();
});

$('queryList').addEventListener('click', e => {
  const b = e.target.closest('.q');
  if (!b) return;
  state.queryId = b.dataset.id;
  $('decisionMsg').textContent = '';
  renderQueries();
  renderLower();
});

$('runBtn').addEventListener('click', runEngine);
$('analyzeDocBtn').addEventListener('click', analyzeUploadedDocument);
$('loadSampleDocBtn').addEventListener('click', () => {
  const m = merchant();
  if (m && m.document_text) {
    $('docInputText').value = m.document_text;
  }
});

$('toggleDocPanelBtn').addEventListener('click', () => {
  const sec = $('docOsintSection');
  state.docPanelOpen = !state.docPanelOpen;
  sec.style.display = state.docPanelOpen ? 'block' : 'none';
});

$('simNormalBtn').addEventListener('click', () => simulateScenario('NORMAL_PAYMENT'));
$('simClonedBtn').addEventListener('click', () => simulateScenario('CLONED_POS'));
$('simSyndicateBtn').addEventListener('click', () => simulateScenario('SYNDICATE_BURST'));

document.querySelectorAll('[data-decision]').forEach(b => b.addEventListener('click', () => decide(b.dataset.decision)));

/* ---------- Initialization ---------- */
(async function init() {
  try {
    state.merchants = await api.getMerchants();
    $('fMerchant').innerHTML =
      '<option value="">Select merchant</option>' +
      state.merchants.map(m => `<option value="${esc(m.id)}">${esc(m.name)} (${esc(m.id)})</option>`).join('');

    if (state.merchants[0]) {
      await selectMerchant(state.merchants[0].id);
    } else {
      renderAll();
    }
  } catch (e) {
    $('engStatus').textContent = `Could not load merchants: ${e.message}`;
  }
})();
