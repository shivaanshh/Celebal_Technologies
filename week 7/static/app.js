/* ── Helpers ────────────────────────────────────────────────────────────── */
const $  = id => document.getElementById(id);
const esc = s  => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

/* ── State ──────────────────────────────────────────────────────────────── */
let pendingFiles       = [];
let isReady            = false;
let conversationHistory = [];   // [{role:"user"|"assistant", content:string}]

/* ── Element refs ───────────────────────────────────────────────────────── */
const dropZone      = $('dropZone');
const fileInput     = $('fileInput');
const browseBtn     = $('browseBtn');
const fileList      = $('fileList');
const backendSelect = $('backendSelect');
const modelSelect   = $('modelSelect');
const topKInput     = $('topKInput');
const topKLabel     = $('topKLabel');
const processBtn    = $('processBtn');
const stageText     = $('stageText');
const activeSection = $('activeSection');
const indexBadge    = $('indexBadge');
const docLog        = $('docLog');
const docSummary    = $('docSummary');
const scText        = $('scText');
const scMeta        = $('scMeta');
const resetBtn      = $('resetBtn');
const messages      = $('messages');
const questionInput = $('questionInput');
const sendBtn       = $('sendBtn');
const hdrDot        = $('hdrDot');
const hdrText       = $('hdrText');
const toastWrap     = $('toastWrap');

/* ── Model lists per backend ────────────────────────────────────────────── */
const MODELS = {
  groq: [
    { v: 'llama-3.3-70b-versatile',  l: 'Llama 3.3 70B Versatile' },
    { v: 'llama-3.1-70b-versatile',  l: 'Llama 3.1 70B Versatile' },
    { v: 'llama-3.1-8b-instant',     l: 'Llama 3.1 8B Instant' },
    { v: 'mixtral-8x7b-32768',       l: 'Mixtral 8x7B' },
    { v: 'gemma2-9b-it',             l: 'Gemma 2 9B' },
    { v: 'llama-3.2-11b-vision-preview', l: 'Llama 3.2 11B Vision' },
  ],
  anthropic: [
    { v: 'claude-haiku-4-5-20251001', l: 'Claude Haiku 4.5' },
    { v: 'claude-sonnet-4-6',         l: 'Claude Sonnet 4.6' },
    { v: 'claude-opus-4-8',           l: 'Claude Opus 4.8' },
  ],
  openai: [
    { v: 'gpt-4o-mini', l: 'GPT-4o Mini' },
    { v: 'gpt-4o',      l: 'GPT-4o' },
    { v: 'gpt-4-turbo', l: 'GPT-4 Turbo' },
  ],
  ollama: [],   // populated at runtime from /api/ollama-models
  auto:  [],
  local: [],
};

async function updateModelDropdown(backend) {
  let options = MODELS[backend] || [];

  if (backend === 'ollama') {
    try {
      const data = await fetch('/api/ollama-models').then(r => r.json());
      options = (data.models || []).map(m => ({ v: m, l: m }));
      MODELS.ollama = options;
    } catch { /* Ollama not running — leave empty */ }
  }

  modelSelect.innerHTML = '';
  if (!options.length) {
    modelSelect.innerHTML = '<option value="">Default</option>';
    modelSelect.disabled = true;
    return;
  }

  modelSelect.disabled = false;
  options.forEach(({ v, l }, i) => {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = l;
    if (i === 0) opt.selected = true;
    modelSelect.appendChild(opt);
  });
}

backendSelect.addEventListener('change', () => updateModelDropdown(backendSelect.value));
updateModelDropdown(backendSelect.value);  // run on page load

/* ── Toast ──────────────────────────────────────────────────────────────── */
function toast(msg, type = 'info', ms = 3500) {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  toastWrap.appendChild(el);
  setTimeout(() => el.remove(), ms);
}

/* ── Utilities ──────────────────────────────────────────────────────────── */
function formatSize(bytes) {
  if (bytes < 1024)        return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function typeClass(filename) {
  const ext = filename.slice(filename.lastIndexOf('.')).toLowerCase();
  return { '.pdf': 'pdf', '.txt': 'txt', '.md': 'md' }[ext] || 'other';
}

function renderDocLog(docStats, totalChunks, backend, model) {
  if (!docStats || !docStats.length) {
    activeSection.style.display = 'none';
    return;
  }
  activeSection.style.display = 'flex';
  indexBadge.textContent = `${docStats.length} doc${docStats.length > 1 ? 's' : ''} · ${totalChunks} chunks`;

  docLog.innerHTML = '';
  docStats.forEach(d => {
    const pct  = totalChunks > 0 ? Math.round(d.chunks / totalChunks * 100) : 0;
    const tc   = typeClass(d.name);
    const label = tc.toUpperCase();
    const row  = document.createElement('div');
    row.className = 'doc-entry';
    row.title = `${d.name} — ${d.chunks} chunks (${pct}% of index)`;
    row.innerHTML = `
      <div class="de-type de-${tc}">${label}</div>
      <div class="de-info">
        <span class="de-name">${esc(d.name)}</span>
        <div class="de-bar"><div class="de-fill" style="width:${pct}%"></div></div>
      </div>
      <span class="de-chunks">${d.chunks} chunks</span>`;
    docLog.appendChild(row);
  });

  docSummary.textContent =
    `Backend: ${backend}  ·  Model: ${model || 'default'}`;
}

/* ── File handling ──────────────────────────────────────────────────────── */
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', e => {
  addFiles([...e.target.files]);
  fileInput.value = '';
});

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('over');
  addFiles([...e.dataTransfer.files]);
});

function addFiles(files) {
  const allowed = ['.pdf', '.txt', '.md'];
  let added = 0;
  files.forEach(f => {
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
    if (!allowed.includes(ext)) { toast(`"${f.name}" skipped — unsupported type.`, 'error'); return; }
    if (pendingFiles.some(p => p.name === f.name)) return;
    pendingFiles.push(f);
    added++;
  });
  if (added) renderFileList();
  processBtn.disabled = pendingFiles.length === 0;
}

function removeFile(name) {
  pendingFiles = pendingFiles.filter(f => f.name !== name);
  renderFileList();
  processBtn.disabled = pendingFiles.length === 0;
}

function renderFileList() {
  fileList.innerHTML = '';
  pendingFiles.forEach(f => {
    const el = document.createElement('div');
    el.className = 'file-item';
    el.innerHTML = `
      <svg class="fi-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      <span class="fi-name" title="${esc(f.name)}">${esc(f.name)}</span>
      <span class="fi-size">${formatSize(f.size)}</span>
      <button class="fi-rm" title="Remove">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>`;
    el.querySelector('.fi-rm').addEventListener('click', e => { e.stopPropagation(); removeFile(f.name); });
    fileList.appendChild(el);
  });
}

/* ── Top-K slider ───────────────────────────────────────────────────────── */
topKInput.addEventListener('input', () => { topKLabel.textContent = topKInput.value; });

/* ── Process documents ──────────────────────────────────────────────────── */
let _pollTimer = null;

function _stopPolling() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

function _showStage(text) {
  stageText.textContent = text;
  stageText.style.display = text ? 'flex' : 'none';
}

async function _pollProgress() {
  try {
    const d = await fetch('/api/ingest-progress').then(r => r.json());

    if (d.status === 'running') {
      _showStage(d.stage);
      return;
    }

    // Terminal states
    _stopPolling();
    _showStage('');
    setProcessing(false);

    if (d.status === 'done') {
      setReady(true, d.result);
      toast(`Loaded ${d.result.files.length} file(s) · ${d.result.chunks} chunks · ${d.result.generation_backend}`, 'success');
      if (d.result.skipped?.length) {
        d.result.skipped.forEach(s => {
          const name = s.name || s;
          const reason = s.reason || 'no extractable text — may be a scanned/image PDF';
          toast(`Skipped "${name}": ${reason}`, 'error', 9000);
        });
      }
    } else if (d.status === 'error') {
      toast(d.error || 'Ingestion failed', 'error', 7000);
    }
  } catch { /* network hiccup — keep polling */ }
}

processBtn.addEventListener('click', async () => {
  if (!pendingFiles.length) return;

  const form = new FormData();
  pendingFiles.forEach(f => form.append('files', f));
  form.append('backend', backendSelect.value);
  form.append('model',   modelSelect.value);

  setProcessing(true);
  _showStage('Starting…');

  try {
    const res = await fetch('/api/ingest', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start ingestion');
    }
    // Start polling every 600 ms
    _pollTimer = setInterval(_pollProgress, 600);
  } catch (e) {
    _stopPolling();
    _showStage('');
    setProcessing(false);
    toast(e.message, 'error', 6000);
  }
});

function setProcessing(on) {
  processBtn.classList.toggle('loading', on);
  processBtn.disabled = on || pendingFiles.length === 0;
}

/* ── Ready state ────────────────────────────────────────────────────────── */
function setReady(ready, data = null) {
  isReady = ready;
  questionInput.disabled = !ready;
  sendBtn.disabled = !ready || !questionInput.value.trim();

  hdrDot.className = `dot ${ready ? 'dot-green' : 'dot-gray'}`;
  hdrText.textContent = ready && data
    ? `${data.files.length} file(s) · ${data.chunks} chunks · ${data.generation_backend}`
    : 'No documents loaded';

  renderDocLog(
    ready && data ? (data.doc_stats || []) : [],
    ready && data ? data.chunks : 0,
    data?.generation_backend || '',
    data?.model || ''
  );

  document.querySelectorAll('.chip').forEach(c => { c.disabled = !ready; });
}

/* ── Reset ──────────────────────────────────────────────────────────────── */
resetBtn.addEventListener('click', async () => {
  await fetch('/api/reset', { method: 'POST' });
  pendingFiles = [];
  conversationHistory = [];
  renderFileList();
  processBtn.disabled = true;
  setReady(false);
  messages.innerHTML = '';
  appendWelcome();
  toast('Reset — ready for new documents.', 'info');
});

/* ── Chat ───────────────────────────────────────────────────────────────── */
questionInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

questionInput.addEventListener('input', () => {
  // auto-resize
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + 'px';
  sendBtn.disabled = !isReady || !questionInput.value.trim();
});

sendBtn.addEventListener('click', send);

async function send() {
  const q = questionInput.value.trim();
  if (!q || !isReady) return;

  questionInput.value = '';
  questionInput.style.height = 'auto';
  sendBtn.disabled = true;

  // remove welcome screen on first message
  const wl = messages.querySelector('.welcome');
  if (wl) wl.remove();

  // render user bubble + typing indicator
  const group = document.createElement('div');
  group.className = 'msg-group';
  group.innerHTML = `<div class="user-msg">${esc(q)}</div>`;

  const aiWrap = document.createElement('div');
  aiWrap.className = 'ai-wrap';
  aiWrap.innerHTML = `<div class="typing"><div class="td"></div><div class="td"></div><div class="td"></div></div>`;
  group.appendChild(aiWrap);
  messages.appendChild(group);
  scrollBottom();

  try {
    const res  = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: q,
        top_k: parseInt(topKInput.value),
        history: conversationHistory.slice(-6),   // last 3 exchanges
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Error');
    renderAnswer(aiWrap, data);

    // Append to conversation history for next turn
    conversationHistory.push({ role: 'user',      content: q });
    conversationHistory.push({ role: 'assistant', content: data.answer });
  } catch (e) {
    aiWrap.innerHTML = `<div class="ai-bubble" style="color:var(--red)">${esc(e.message)}</div>`;
  }

  scrollBottom();
  if (isReady) sendBtn.disabled = !questionInput.value.trim();
}

/* ── Render answer ──────────────────────────────────────────────────────── */
function renderAnswer(container, { answer, sources }) {
  const bubble = document.createElement('div');
  bubble.className = 'ai-bubble';
  bubble.innerHTML = formatAnswer(answer);

  container.innerHTML = '';
  container.appendChild(bubble);

  if (sources && sources.length) {
    container.appendChild(buildSources(sources));
  }
}

function formatAnswer(text) {
  return '<p>' + esc(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\[(\d+)\]/g, '<span style="color:var(--accent);font-weight:600">[$1]</span>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>') + '</p>';
}

function buildSources(sources) {
  const wrap   = document.createElement('div');
  wrap.className = 'sources-wrap';

  const toggle = document.createElement('button');
  toggle.className = 'src-toggle';
  toggle.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
      <polyline points="9 18 15 12 9 6"/>
    </svg>
    ${sources.length} source${sources.length > 1 ? 's' : ''}`;

  const list = document.createElement('div');
  list.className = 'src-list';

  sources.forEach((s, i) => {
    const pct   = Math.round(s.score * 100);
    const cls   = pct >= 50 ? 'score-hi' : pct >= 25 ? 'score-med' : 'score-lo';
    const card  = document.createElement('div');
    card.className = 'src-card';
    card.innerHTML = `
      <div class="src-hdr">
        <span class="src-name">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          [${i+1}] ${esc(s.source)}
        </span>
        <span class="score ${cls}">${pct}%</span>
      </div>
      <div class="src-text">${esc(s.text)}</div>`;
    list.appendChild(card);
  });

  toggle.addEventListener('click', () => {
    toggle.classList.toggle('open');
    list.classList.toggle('open');
  });

  wrap.appendChild(toggle);
  wrap.appendChild(list);
  return wrap;
}

/* ── Scroll ─────────────────────────────────────────────────────────────── */
function scrollBottom() {
  messages.scrollTo({ top: messages.scrollHeight, behavior: 'smooth' });
}

/* ── Welcome screen ─────────────────────────────────────────────────────── */
function appendWelcome() {
  const el = document.createElement('div');
  el.className = 'welcome';
  el.innerHTML = `
    <div class="wl-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <circle cx="11" cy="11" r="8"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
    </div>
    <h2 class="wl-title">Ask anything about your documents</h2>
    <p class="wl-sub">Upload files on the left, hit <strong>Process Documents</strong>, then ask questions here.</p>
    <div class="chips">
      <button class="chip" data-q="What is the main idea of this document?">Main idea</button>
      <button class="chip" data-q="Summarize the key points.">Key points</button>
      <button class="chip" data-q="What are the most important conclusions?">Conclusions</button>
      <button class="chip" data-q="What limitations or challenges are mentioned?">Limitations</button>
    </div>`;
  wireChips(el);
  messages.appendChild(el);
}

function wireChips(root) {
  root.querySelectorAll('.chip').forEach(chip => {
    chip.disabled = !isReady;
    chip.addEventListener('click', () => {
      if (!isReady) return;
      questionInput.value = chip.dataset.q;
      send();
    });
  });
}

/* ── Init ───────────────────────────────────────────────────────────────── */
wireChips(document);   // wire the static welcome chips in the HTML

// restore server state on page refresh
fetch('/api/status')
  .then(r => r.json())
  .then(d => {
    if (d.ready) setReady(true, {
      files: d.files, chunks: d.chunks,
      doc_stats: d.doc_stats,
      generation_backend: d.generation_backend, model: d.model,
    });
  })
  .catch(() => {});
