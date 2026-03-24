'use strict';

/* ─── Class Definitions ───────────────────────────────────────────────────── */
const CLASSES = [
  {
    label:  'Clear Skin',
    color:  '#8a9e8c',
    badge:  'No Condition',
    desc:   'No significant aging condition detected. Skin appears healthy.',
    tips: [
      { icon: '✅', text: 'Keep applying SPF daily.' },
      { icon: '🌿', text: 'Maintain a simple consistent routine.' },
      { icon: '🔁', text: 'Re-scan periodically.' },
    ]
  },
  {
    label:  'Dark Spots',
    color:  '#9d8189',
    badge:  'Hyperpigmentation',
    desc:   'Dark spots result from excess melanin due to UV exposure or hormonal changes...',
    tips: [
      { icon: '✨', text: 'Use Vitamin C or niacinamide serum daily.' },
      { icon: '🧪', text: 'Exfoliate 2–3x a week with AHA/BHA.' },
      { icon: '🚫', text: 'Avoid picking at blemishes.' },
    ]
  },
  {
    label:  'Puffy Eyes',
    color:  '#c08b96',
    badge:  'Periorbital',
    desc:   'Fluid accumulation under eyes caused by sleep, allergies, or aging.',
    tips: [
      { icon: '😴', text: 'Prioritize 7–9 hours of sleep.' },
      { icon: '❄️', text: 'Use cold compresses in the morning.' },
      { icon: '🧂', text: 'Reduce salt intake.' },
    ]
  },
  {
    label:  'Wrinkles',
    color:  '#b5838d',
    badge:  'Aging Sign',
    desc:   'Fine lines and wrinkles form when the skin loses elasticity and collagen over time...',
    tips: [
      { icon: '☀️', text: 'Apply broad-spectrum SPF 30+ every morning.' },
      { icon: '🧴', text: 'Use a retinol or peptide serum at night.' },
      { icon: '💧', text: 'St ay well-hydrated and moisturize.' },
    ]
  },
];

/* ─── DOM References ──────────────────────────────────────────────────────── */
const uploadBtn      = document.getElementById('uploadBtn');
const fileInput      = document.getElementById('fileInput');
const resultsSection = document.getElementById('resultsSection');
const emptyState     = document.getElementById('emptyState');
const uploadedImg    = document.getElementById('uploadedImg');
const resultBadge    = document.getElementById('resultBadge');
const resultClass    = document.getElementById('resultClass');
const confVal        = document.getElementById('confVal');
const confFill       = document.getElementById('confFill');
const barChart       = document.getElementById('barChart');
const conditionChip  = document.getElementById('conditionChip');
const conditionDesc  = document.getElementById('conditionDesc');
const tipsList       = document.getElementById('tipsList');
const imageNote      = document.getElementById('imageNote');
const CLASS_NAMES    = CLASSES.map(c => c.label);

/* ─── Event Listeners ─────────────────────────────────────────────────────── */
uploadBtn.addEventListener('click', (e) => {
  e.preventDefault();
  fileInput.click();
});
fileInput.addEventListener('change', (e) => {
  e.preventDefault();
  if (fileInput.files[0]) run(fileInput.files[0]);
  fileInput.value = '';
});
document.addEventListener('dragover', e => { e.preventDefault(); document.body.classList.add('drag-active'); });
document.addEventListener('dragleave', e => { if (e.relatedTarget === null) document.body.classList.remove('drag-active'); });
document.addEventListener('drop', e => {
  e.preventDefault();
  document.body.classList.remove('drag-active');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) run(f);
});

/* ─── Core Pipeline ───────────────────────────────────────────────────────── */

/**
 * Main entry point: upload file, get prediction, render UI
 */
async function run(file) {
  // ✅ Revoke old object URL to force image refresh
  if (uploadedImg.src.startsWith('blob:')) {
    URL.revokeObjectURL(uploadedImg.src);
  }
  uploadedImg.src = URL.createObjectURL(file);

  // ✅ Reset UI state before fetch
  resultsSection.hidden = true;
  emptyState.style.display = 'flex';
  barChart.innerHTML = '';
  tipsList.innerHTML = '';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error('Server error');
    const data = await res.json();

    const preds = CLASSES.map(c => {
      const idx = CLASS_NAMES.indexOf(c.label);
      const probRaw = idx >= 0 && data.all_probs[idx] != null ? data.all_probs[idx] : 0;
      return {
        ...c,
        prob: Math.round(probRaw * 100),
        top: c.label.toLowerCase() === (data.class || '').toLowerCase()
      };
    });

    render(preds, file);

  } catch (err) {
    alert('Could not get prediction from server.');
    console.error(err);
  }
}

/**
 * Call FastAPI backend for prediction
 */
async function predict(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://127.0.0.1:8000/predict', { method: 'POST', body: formData });
  if (!response.ok) throw new Error('Server error');

  const data = await response.json();

  console.log('Backend response:', data); // check this in dev console

  // Map backend response to CLASSES safely
  return CLASSES.map(c => {
    const idx = CLASS_NAMES.indexOf(c.label);
    const probRaw = idx >= 0 && data.all_probs[idx] != null ? data.all_probs[idx] : 0;
    return {
      ...c,
      prob: Math.round(probRaw * 100), // now 0.87 -> 87%
      top: c.label.toLowerCase() === (data.class || '').toLowerCase()
    };
  });
}

/**
 * Render prediction results in the UI
 */
function render(preds, file) {
  const top = preds.find(p => p.top);
  if (!top) return;

  emptyState.style.display = 'none';
  resultsSection.hidden = false;

  // Badge + class
  resultBadge.textContent = top.badge;
  resultClass.textContent = top.label;
  resultClass.style.color = top.color;

  // Confidence
  confVal.textContent = top.prob + '%';
  confFill.style.width = top.prob + '%';
  confFill.style.background = top.color;

  // File info
  imageNote.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;

  // Bar chart
  barChart.innerHTML = '';
  [...preds].sort((a, b) => b.prob - a.prob).forEach(p => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-meta">
        <span class="bar-name ${p.top ? 'top' : ''}">${p.label}</span>
        <span class="bar-pct ${p.top ? 'top' : ''}">${p.prob}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${p.prob}%; background:${p.color};"></div>
      </div>`;
    barChart.appendChild(row);
  });

  // Condition info
  conditionChip.textContent = top.label;
  conditionDesc.textContent = top.desc;

  // Tips
  tipsList.innerHTML = top.tips.map(t => `
    <div class="tip-row">
      <span class="tip-icon">${t.icon}</span>
      <span class="tip-text">${t.text}</span>
    </div>`).join('');

  //window.scrollTo({ top: 0, behavior: 'smooth' });
}