const state = {
  model: document.getElementById('modelSelect').value,
  engines: [],
  selectedId: null,
  chart: null,
};

const els = {
  modelSelect: document.getElementById('modelSelect'),
  fleetGrid: document.getElementById('fleetGrid'),
  fleetSub: document.getElementById('fleetSub'),
  statTotal: document.getElementById('statTotal'),
  statCritical: document.getElementById('statCritical'),
  statWarning: document.getElementById('statWarning'),
  statHealthy: document.getElementById('statHealthy'),
  statAvg: document.getElementById('statAvg'),
  detailEmpty: document.getElementById('detailEmpty'),
  detailContent: document.getElementById('detailContent'),
  detailTitle: document.getElementById('detailTitle'),
  detailBadge: document.getElementById('detailBadge'),
  gaugeArc: document.getElementById('gaugeArc'),
  gaugeNeedle: document.getElementById('gaugeNeedle'),
  gaugeValue: document.getElementById('gaugeValue'),
  metaPred: document.getElementById('metaPred'),
  metaTrue: document.getElementById('metaTrue'),
  metaError: document.getElementById('metaError'),
  metaModel: document.getElementById('metaModel'),
};

const GAUGE_MAX = 125; // matches RUL_CAP used in training

function statusFromRul(rul) {
  if (rul < 20) return 'critical';
  if (rul < 50) return 'warning';
  return 'healthy';
}

async function loadFleet() {
  els.fleetSub.textContent = 'loading…';
  const res = await fetch(`/api/predict_all?model=${state.model}`);
  const data = await res.json();
  state.engines = data;
  renderFleet();
  renderStats();
}

function renderStats() {
  const total = state.engines.length;
  const critical = state.engines.filter(e => e.status === 'critical').length;
  const warning = state.engines.filter(e => e.status === 'warning').length;
  const healthy = state.engines.filter(e => e.status === 'healthy').length;
  const avg = state.engines.reduce((s, e) => s + e.predicted_rul, 0) / (total || 1);

  els.statTotal.textContent = total;
  els.statCritical.textContent = critical;
  els.statWarning.textContent = warning;
  els.statHealthy.textContent = healthy;
  els.statAvg.textContent = avg.toFixed(1);
  els.fleetSub.textContent = `${total} engines · ${state.model.toUpperCase()} model`;
}

function renderFleet() {
  els.fleetGrid.innerHTML = '';
  const sorted = [...state.engines].sort((a, b) => a.predicted_rul - b.predicted_rul);
  for (const engine of sorted) {
    const card = document.createElement('div');
    card.className = `engine-card status-${engine.status}`;
    card.tabIndex = 0;
    card.dataset.id = engine.engine_id;
    if (engine.engine_id === state.selectedId) card.classList.add('selected');
    card.innerHTML = `
      <div class="status-dot"></div>
      <div class="eid">ENGINE #${engine.engine_id}</div>
      <div class="rul">${engine.predicted_rul}</div>
    `;
    card.addEventListener('click', () => selectEngine(engine.engine_id));
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') selectEngine(engine.engine_id);
    });
    els.fleetGrid.appendChild(card);
  }
}

async function selectEngine(id) {
  state.selectedId = id;
  renderFleet();

  const res = await fetch(`/api/engine/${id}?model=${state.model}`);
  const data = await res.json();
  if (data.error) return;

  els.detailEmpty.hidden = true;
  els.detailContent.hidden = false;

  els.detailTitle.textContent = `Engine #${data.engine_id}`;
  els.detailBadge.textContent = data.status;
  els.detailBadge.className = `badge status-${data.status}`;

  updateGauge(data.predicted_rul, data.status);

  els.metaPred.textContent = `${data.predicted_rul} cycles`;
  els.metaTrue.textContent = `${data.true_rul} cycles`;
  els.metaError.textContent = `${Math.abs(data.predicted_rul - data.true_rul).toFixed(1)} cycles`;
  els.metaModel.textContent = data.model_used.toUpperCase();

  renderChart(data);
}

function updateGauge(value, status) {
  const clamped = Math.max(0, Math.min(GAUGE_MAX, value));
  const frac = clamped / GAUGE_MAX;

  // Arc: 0..1 maps to angle -90deg..+90deg along the semicircle path length
  const arcLength = Math.PI * 80; // r=80 semicircle
  els.gaugeArc.style.strokeDasharray = `${arcLength}`;
  els.gaugeArc.style.strokeDashoffset = `${arcLength * (1 - frac)}`;

  const colorVar = status === 'critical' ? '--critical' : status === 'warning' ? '--warn' : '--safe';
  els.gaugeArc.style.stroke = getComputedStyle(document.documentElement).getPropertyValue(colorVar);

  const angleDeg = -90 + frac * 180;
  els.gaugeNeedle.style.transform = `rotate(${angleDeg}deg)`;

  els.gaugeValue.textContent = value;
  els.gaugeValue.style.color = els.gaugeArc.style.stroke;
}

function renderChart(data) {
  const ctx = document.getElementById('sensorChart').getContext('2d');
  const colors = ['#4EA1FF', '#38D68C', '#F5A623', '#FF5A5F', '#B98CFF'];
  const datasets = Object.entries(data.sensor_traces).map(([name, values], i) => ({
    label: name.replace('_', ' '),
    data: values,
    borderColor: colors[i % colors.length],
    borderWidth: 1.75,
    pointRadius: 0,
    tension: 0.25,
    yAxisID: `y${i % 2}`,
  }));

  if (state.chart) state.chart.destroy();
  state.chart = new Chart(ctx, {
    type: 'line',
    data: { labels: data.cycles, datasets },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#8A97A3', font: { family: 'IBM Plex Mono', size: 11 } },
        },
      },
      scales: {
        x: {
          ticks: { color: '#8A97A3', font: { family: 'IBM Plex Mono', size: 10 }, maxTicksLimit: 8 },
          grid: { color: '#2B3540' },
          title: { display: true, text: 'cycle', color: '#8A97A3' },
        },
        y0: {
          position: 'left',
          ticks: { color: '#8A97A3', font: { size: 10 } },
          grid: { color: '#2B3540' },
        },
        y1: {
          position: 'right',
          ticks: { color: '#8A97A3', font: { size: 10 } },
          grid: { display: false },
        },
      },
    },
  });
}

els.modelSelect.addEventListener('change', () => {
  state.model = els.modelSelect.value;
  loadFleet();
  if (state.selectedId) selectEngine(state.selectedId);
});

loadFleet();
