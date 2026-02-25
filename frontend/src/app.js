const API_URL = 'http://localhost:8000';

// Navigation
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const page = e.target.dataset.page;
    
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    
    e.target.classList.add('active');
    document.getElementById(`${page}-page`).classList.add('active');
    
    if (page === 'history') loadHistory();
    if (page === 'analytics') loadAnalytics();
  });
});

// File Upload
const imageInput = document.getElementById('image-input');
const analyzeBtn = document.getElementById('analyze-btn');
const loading = document.getElementById('loading');
const result = document.getElementById('result');

imageInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    analyzeBtn.disabled = false;
    const fileName = e.target.files[0].name;
    const maxLength = 25;
    const displayName = fileName.length > maxLength ? fileName.substring(0, maxLength) + '...' : fileName;
    analyzeBtn.textContent = `Analyze ${displayName}`;
  }
});

analyzeBtn.addEventListener('click', async () => {
  const file = imageInput.files[0];
  if (!file) return;
  
  loading.classList.remove('hidden');
  result.classList.add('hidden');
  
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    console.log('Connecting to:', `${API_URL}/api/analyze`);
    const response = await fetch(`${API_URL}/api/analyze`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    displayResults(data);
  } catch (error) {
    alert(`Error analyzing image: ${error.message}. Make sure the backend is running on ${API_URL}`);
    console.error('Full error:', error);
  } finally {
    loading.classList.add('hidden');
  }
});

let currentResult = null;
let showingLabels = true;

function displayResults(data) {
  currentResult = data;
  showingLabels = true;
  
  // Image and basic info
  const annotatedImg = document.getElementById('annotated');
  const verdictEl = document.getElementById('verdict');
  
  if (annotatedImg) annotatedImg.src = data.annotated;
  if (verdictEl) verdictEl.textContent = data.verdict;
  
  // Detection summary (horizontal)
  const breadCountEl = document.getElementById('bread-count');
  const moldCountEl = document.getElementById('mold-count');
  const riskEl = document.getElementById('risk');
  
  if (breadCountEl) breadCountEl.textContent = data.detections_count.bread;
  if (moldCountEl) moldCountEl.textContent = data.detections_count.mold;
  if (riskEl) riskEl.textContent = data.risk;
  
  // Coverage bar
  const coverageFill = document.getElementById('coverage-fill');
  const coverageText = document.getElementById('coverage-text');
  
  if (coverageFill) coverageFill.style.width = `${data.coverage}%`;
  if (coverageText) coverageText.textContent = `${data.coverage}%`;
  
  // Verdict card details
  const verdictStatus = document.getElementById('verdict-status');
  const riskDisplay = document.getElementById('risk-display');
  const coverageDisplay = document.getElementById('coverage-display');
  
  if (verdictStatus) verdictStatus.textContent = data.verdict;
  if (riskDisplay) riskDisplay.textContent = data.risk;
  if (coverageDisplay) coverageDisplay.textContent = `${data.coverage}%`;
  
  // Result breakdown
  const breadTypeEl = document.getElementById('bread-type');
  const moldTypeEl = document.getElementById('mold-type');
  const storageTimeEl = document.getElementById('storage-time');
  const breadAgeEl = document.getElementById('bread-age');
  const ageDaysEl = document.getElementById('age-days');
  
  if (breadTypeEl) breadTypeEl.textContent = data.bread_type;
  if (moldTypeEl) moldTypeEl.textContent = data.mold_type;
  if (storageTimeEl) storageTimeEl.textContent = data.storage_time;
  if (breadAgeEl) breadAgeEl.textContent = data.bread_age;
  if (ageDaysEl) ageDaysEl.textContent = data.age_days;
  
  // Mold information section
  const moldInfoSection = document.getElementById('mold-info-section');
  const moldInfoList = document.getElementById('mold-information-list');
  if (data.mold_type !== 'None' && data.coverage > 0) {
    const moldDescriptions = {
      'Mold Aspergillus': 'Black or dark-colored mold that produces mycotoxins. Forms in warm, humid conditions and appears as black, brown, or greenish-black spots.',
      'Mold Cladosporium': 'Olive-green to brown mold commonly found on bread. Develops in moist environments and appears as dark green to black velvety patches.',
      'Mold Penicillium': 'Blue-green mold, the most common bread mold species. Grows rapidly in humid conditions and appears as blue, green, or white fuzzy growth.',
      'Mold Rhizopus': 'Black bread mold that grows quickly in warm conditions. Starts as white fuzzy growth that turns black as spores develop.'
    };
    
    const moldDesc = moldDescriptions[data.mold_type] || 'Fungal growth that develops on bread when exposed to moisture and warmth. Mold spores can penetrate deep into the bread\'s porous structure.';
    
    moldInfoList.innerHTML = `
      <li><strong>Mold Type:</strong> ${data.mold_type}</li>
      <li><strong>Coverage:</strong> ${data.coverage}% of bread surface affected</li>
      <li><strong>Description:</strong> ${moldDesc}</li>
    `;
    moldInfoSection.classList.remove('hidden');
  } else {
    moldInfoSection.classList.add('hidden');
  }
  
  // Actions
  const actionEl = document.getElementById('action');
  if (actionEl) actionEl.textContent = data.action;
  
  // Verdict badge styling
  const verdictBadge = document.getElementById('verdict-badge');
  verdictBadge.className = 'verdict-badge';
  verdictBadge.classList.add(data.verdict === 'Healthy' ? 'healthy' : 'unhealthy');
  
  // Toggle labels button
  document.getElementById('toggle-labels').textContent = 'Hide Labels';
  
  result.classList.remove('hidden');
  result.scrollIntoView({ behavior: 'smooth' });
}

document.getElementById('toggle-labels')?.addEventListener('click', () => {
  if (!currentResult) return;
  
  showingLabels = !showingLabels;
  const img = document.getElementById('annotated');
  const btn = document.getElementById('toggle-labels');
  
  if (showingLabels) {
    img.src = currentResult.annotated;
    btn.textContent = 'Hide Labels';
  } else {
    img.src = currentResult.annotated_no_labels;
    btn.textContent = 'Show Labels';
  }
});

async function loadHistory() {
  try {
    const response = await fetch(`${API_URL}/api/history`);
    const history = await response.json();
    
    const historyList = document.getElementById('history-list');
    
    if (history.length === 0) {
      historyList.innerHTML = '<p style="text-align: center; color: var(--text-light);">No analysis history yet</p>';
      return;
    }
    
    historyList.innerHTML = history.reverse().map(item => `
      <div class="history-item">
        <img src="${item.annotated}" alt="Analysis">
        <div class="history-info">
          <div class="timestamp">${new Date(item.timestamp).toLocaleString()}</div>
          <div class="verdict" style="color: ${item.verdict === 'Healthy' ? 'var(--success)' : 'var(--danger)'}">${item.verdict}</div>
          <div style="font-size: 0.9rem;">Risk: ${item.risk} | Coverage: ${item.coverage}%</div>
          <div style="font-size: 0.85rem; color: var(--text-light);">Bread: ${item.bread_type} | Mold: ${item.mold_type}</div>
        </div>
        <div style="text-align: right;">
          <div style="font-weight: 600; color: var(--primary);">${item.bread_age}</div>
          <div style="font-size: 0.9rem; color: var(--text-light);">${item.storage_time}</div>
        </div>
      </div>
    `).join('');
  } catch (error) {
    console.error('Error loading history:', error);
  }
}

let charts = {};

function destroyCharts() {
  Object.values(charts).forEach(chart => chart?.destroy());
  charts = {};
}

async function loadAnalytics() {
  try {
    const response = await fetch(`${API_URL}/api/analytics`);
    const analytics = await response.json();
    
    const analyticsStats = document.getElementById('analytics-stats');
    
    if (analytics.total_scans === 0) {
      analyticsStats.innerHTML = '<p style="text-align: center; color: var(--text-light); grid-column: 1/-1;">No analytics data yet</p>';
      destroyCharts();
      return;
    }
    
    analyticsStats.innerHTML = `
      <div class="stat-card">
        <div class="stat-value">${analytics.total_scans}</div>
        <div class="stat-label">Total Scans</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color: var(--success);">${analytics.healthy}</div>
        <div class="stat-label">Healthy</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color: var(--danger);">${analytics.unhealthy}</div>
        <div class="stat-label">Unhealthy</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${analytics.avg_coverage}%</div>
        <div class="stat-label">Avg Coverage</div>
      </div>
    `;
    
    destroyCharts();
    createCharts(analytics);
  } catch (error) {
    console.error('Error loading analytics:', error);
  }
}

function createCharts(analytics) {
  const chartColors = {
    primary: '#8B5A3C',
    secondary: '#A0826D',
    accent: '#C9A88A',
    success: '#7CB342',
    danger: '#E53935',
    warning: '#FFA726',
    pastelBlue: '#A8D5E2',
    pastelOrange: '#FFB88C',
    pastelBlueDark: '#7FB3D5',
    pastelOrangeDark: '#FF9A6C'
  };
  
  // Risk Distribution Chart
  const riskCtx = document.getElementById('riskChart');
  if (riskCtx) {
    charts.risk = new Chart(riskCtx, {
      type: 'doughnut',
      data: {
        labels: Object.keys(analytics.risk_distribution),
        datasets: [{
          data: Object.values(analytics.risk_distribution),
          backgroundColor: [chartColors.pastelBlue, chartColors.pastelOrange, chartColors.pastelOrangeDark, chartColors.pastelBlueDark]
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { position: 'bottom' }
        }
      }
    });
  }
  
  // Verdict Chart
  const verdictCtx = document.getElementById('verdictChart');
  if (verdictCtx) {
    charts.verdict = new Chart(verdictCtx, {
      type: 'pie',
      data: {
        labels: ['Healthy', 'Unhealthy'],
        datasets: [{
          data: [analytics.healthy, analytics.unhealthy],
          backgroundColor: [chartColors.pastelBlue, chartColors.pastelOrange]
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { position: 'bottom' }
        }
      }
    });
  }
  
  // Bread Types Chart
  const breadCtx = document.getElementById('breadChart');
  if (breadCtx && Object.keys(analytics.bread_types).length > 0) {
    charts.bread = new Chart(breadCtx, {
      type: 'bar',
      data: {
        labels: Object.keys(analytics.bread_types),
        datasets: [{
          label: 'Count',
          data: Object.values(analytics.bread_types),
          backgroundColor: chartColors.pastelBlue
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: { beginAtZero: true, ticks: { stepSize: 1 } }
        }
      }
    });
  }
  
  // Mold Types Chart
  const moldCtx = document.getElementById('moldChart');
  if (moldCtx && Object.keys(analytics.mold_types).length > 0) {
    charts.mold = new Chart(moldCtx, {
      type: 'bar',
      data: {
        labels: Object.keys(analytics.mold_types),
        datasets: [{
          label: 'Count',
          data: Object.values(analytics.mold_types),
          backgroundColor: chartColors.pastelOrange
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: { beginAtZero: true, ticks: { stepSize: 1 } }
        }
      }
    });
  }
  
  // Age Distribution Chart
  const ageCtx = document.getElementById('ageChart');
  if (ageCtx && Object.keys(analytics.age_distribution).length > 0) {
    charts.age = new Chart(ageCtx, {
      type: 'bar',
      data: {
        labels: Object.keys(analytics.age_distribution),
        datasets: [{
          label: 'Count',
          data: Object.values(analytics.age_distribution),
          backgroundColor: [chartColors.pastelBlue, chartColors.pastelBlueDark, chartColors.pastelOrange, chartColors.pastelOrangeDark]
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: { beginAtZero: true, ticks: { stepSize: 1 } }
        }
      }
    });
  }
}

document.getElementById('clear-history')?.addEventListener('click', async () => {
  if (confirm('Are you sure you want to clear all history?')) {
    // This would need a backend endpoint to clear history
    alert('Clear history feature - implement backend endpoint');
  }
});
