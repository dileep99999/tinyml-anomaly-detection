// ── Tab switching ────────────────────────────────────────────────
function switchTab(name, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');

  const schematic = document.getElementById('deploy-schematic-card');
  const resultCard = document.getElementById('result-card');
  const placeholder = document.getElementById('placeholder');
  
  if (name === 'deploy') {
    schematic.classList.remove('hidden');
    resultCard.classList.add('hidden');
    placeholder.classList.add('hidden');
  } else {
    schematic.classList.add('hidden');
    if (isStreaming) {
      resultCard.classList.remove('hidden');
      placeholder.classList.add('hidden');
    } else if (resultCard.querySelector('#verdict-text') && resultCard.querySelector('#verdict-text').textContent !== "") {
      resultCard.classList.remove('hidden');
      placeholder.classList.add('hidden');
    } else {
      placeholder.classList.remove('hidden');
    }
  }
}

// ── File selection ───────────────────────────────────────────────
function fileSelected(input) {
  const name = input.files[0] ? input.files[0].name : 'No file selected';
  document.getElementById('file-name').textContent = name;
}

// Drag and drop
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#3949ab'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = '#2a2a4a'; });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.style.borderColor = '#2a2a4a';
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.mat')) {
    document.getElementById('file-input').files = e.dataTransfer.files;
    document.getElementById('file-name').textContent = file.name;
  }
});

// ── Show loading / result helpers ────────────────────────────────
function showLoading() {
  document.getElementById('placeholder').classList.add('hidden');
  document.getElementById('result-card').classList.add('hidden');
  document.getElementById('loading').classList.remove('hidden');
}

function showResult(data) {
  document.getElementById('loading').classList.add('hidden');
  document.getElementById('placeholder').classList.add('hidden');

  const card    = document.getElementById('result-card');
  const banner  = document.getElementById('verdict-banner');
  const isFault = data.is_fault;

  // verdict banner
  banner.className = 'verdict-banner ' + (isFault ? 'fault' : 'normal');
  document.getElementById('verdict-icon').textContent = isFault ? '🔴' : '🟢';
  document.getElementById('verdict-text').textContent = isFault ? 'ANOMALY DETECTED' : 'NORMAL OPERATION';
  document.getElementById('verdict-class').textContent = 'Class: ' + data.class;

  // metrics
  document.getElementById('score-val').textContent   = data.score;
  document.getElementById('thresh-val').textContent  = data.threshold;
  document.getElementById('latency-val').textContent = data.latency + ' ms';
  document.getElementById('windows-val').textContent = data.n_fault + ' / ' + data.n_total;

  // plots
  document.getElementById('recon-plot').src = 'data:image/png;base64,' + data.plot;
  document.getElementById('slide-plot').src = 'data:image/png;base64,' + data.slide;

  card.classList.remove('hidden');
}

function showError(msg) {
  document.getElementById('loading').classList.add('hidden');
  document.getElementById('placeholder').classList.remove('hidden');
  document.getElementById('placeholder').innerHTML =
    '<span>⚠️</span><p style="color:#ef5350">' + msg + '</p>';
}

// ── Predict by class ─────────────────────────────────────────────
function predictClass() {
  const cls = document.getElementById('class-select').value;
  showLoading();
  fetch('/predict_class', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({class_name: cls})
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { showError(data.error); return; }
    showResult(data);
  })
  .catch(() => showError('Server error — is the Flask app running?'));
}

// ── Predict by upload ────────────────────────────────────────────
function predictUpload() {
  const fileInput = document.getElementById('file-input');
  if (!fileInput.files[0]) { alert('Please select a .mat file first'); return; }

  const form = new FormData();
  form.append('file', fileInput.files[0]);
  showLoading();

  fetch('/predict_upload', { method: 'POST', body: form })
  .then(r => r.json())
  .then(data => {
    if (data.error) { showError(data.error); return; }
    showResult(data);
  })
  .catch(() => showError('Server error'));
}

// ── All classes ──────────────────────────────────────────────────
function predictAll() {
  const container = document.getElementById('all-results-table');
  container.innerHTML = '<p style="color:#7986cb;font-size:13px">Running inference on all classes...</p>';

  fetch('/all_classes')
  .then(r => r.json())
  .then(data => {
    const results  = data.results;
    const correct  = results.filter(r => r.correct).length;
    const accuracy = ((correct / results.length) * 100).toFixed(0);

    let html = `<div class="accuracy-bar">Accuracy: ${correct}/${results.length} = ${accuracy}%</div>`;
    html += '<table class="results-table"><thead><tr>';
    html += '<th>Class</th><th>Score</th><th>Decision</th><th>Correct?</th></tr></thead><tbody>';

    results.forEach(r => {
      const decBadge = r.is_fault
        ? '<span class="badge fault">FAULT</span>'
        : '<span class="badge normal">NORMAL</span>';
      const okBadge  = r.correct
        ? '<span class="badge ok">✅</span>'
        : '<span class="badge wrong">❌</span>';
      html += `<tr>
        <td>${r.class}</td>
        <td>${r.score}</td>
        <td>${decBadge}</td>
        <td>${okBadge}</td>
      </tr>`;
    });

    html += '</tbody></table>';
    html += '</tbody></table>';
    container.innerHTML = html;
  })
  .catch(() => {
    container.innerHTML = '<p style="color:#ef5350">Error — is the Flask app running?</p>';
  });
}

// ── Live Stream Simulator & Audio Synthesizer ─────────────────────
let isStreaming = false;
let streamIndex = 0;
let streamClass = 'Normal';
let streamTimer = null;
let streamStepCount = 0;
let sirenCtx = null;
let sirenOsc = null;
let sirenGain = null;

function playSiren() {
  if (!document.getElementById("siren-sound").checked) return;
  
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    
    osc.type = "sawtooth";
    // Sweeping frequency
    osc.frequency.setValueAtTime(500, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(1000, ctx.currentTime + 0.3);
    osc.frequency.exponentialRampToValueAtTime(500, ctx.currentTime + 0.6);
    
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.6);
    
    osc.connect(gain);
    gain.connect(ctx.destination);
    
    osc.start();
    osc.stop(ctx.currentTime + 0.6);
  } catch (e) {
    console.error("Audio context error:", e);
  }
}

function toggleLiveStream() {
  const btn = document.getElementById("btn-stream-toggle");
  
  if (!isStreaming) {
    isStreaming = true;
    streamIndex = 0;
    streamClass = 'Normal';
    streamStepCount = 0;
    document.getElementById("stream-class-select").value = 'Normal';
    
    btn.textContent = "⏹ Stop Stream";
    btn.style.background = "#dc2626";
    btn.style.borderColor = "#b91c1c";
    
    // Clear plots and show placeholder/verdict
    document.getElementById("placeholder").classList.add("hidden");
    document.getElementById("loading").classList.add("hidden");
    document.getElementById("recon-plot").classList.add("hidden");
    document.getElementById("slide-plot").classList.add("hidden");
    document.querySelector(".plot-title").style.display = "none";
    document.querySelectorAll(".plot-title")[1].style.display = "none";
    
    document.getElementById("result-card").classList.remove("hidden");
    updateStreamUIStatus(false, "Initializing stream...");
    
    streamTimer = setInterval(streamStep, 1000);
  } else {
    isStreaming = false;
    clearInterval(streamTimer);
    
    btn.textContent = "▶ Start Stream";
    btn.style.background = "";
    btn.style.borderColor = "";
    
    // Reset view
    document.getElementById("result-card").classList.add("hidden");
    document.getElementById("placeholder").classList.remove("hidden");
    document.getElementById("recon-plot").classList.remove("hidden");
    document.getElementById("slide-plot").classList.remove("hidden");
    document.querySelector(".plot-title").style.display = "block";
    document.querySelectorAll(".plot-title")[1].style.display = "block";
  }
}

function injectStreamFault() {
  if (!isStreaming) {
    alert("Please click 'Start Stream' first before injecting faults!");
    return;
  }
  const selected = document.getElementById("stream-class-select").value;
  streamClass = selected;
  console.log("Injected fault class:", streamClass);
}

function streamStep() {
  streamStepCount++;
  
  // Auto-inject fault after 10 steps (10 seconds)
  if (streamClass === 'Normal' && streamStepCount >= 10) {
    const selected = document.getElementById("stream-class-select").value;
    // Inject the selected anomaly (if Normal is selected, fallback to InnerRace_014)
    streamClass = (selected === 'Normal') ? 'InnerRace_014' : selected;
    console.log("Auto-injecting fault:", streamClass);
  }

  fetch('/stream_data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ class_name: streamClass, index: streamIndex })
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) {
      console.error(data.error);
      toggleLiveStream();
      return;
    }
    
    streamIndex = data.index + 1;
    
    // Update metric cards
    document.getElementById('score-val').textContent = data.score;
    document.getElementById('thresh-val').textContent = data.threshold;
    document.getElementById('latency-val').textContent = data.latency + ' ms';
    document.getElementById('windows-val').textContent = 'Stream Window #' + streamIndex;
    
    // Update banner & play sound if fault
    updateStreamUIStatus(data.is_fault, streamClass);
    if (data.is_fault) {
      playSiren();
      
      // Pause streaming loop while alert blocks
      clearInterval(streamTimer);
      
      setTimeout(() => {
        alert("🚨 ANOMALY DETECTED!\n\nSignal reconstruction MSE score (" + data.score + ") exceeded the safety limit of " + data.threshold + ".\n\nClick OK to reset the system back to Normal operation.");
        
        // Reset state
        streamClass = 'Normal';
        streamStepCount = 0;
        updateStreamUIStatus(false, 'Normal');
        
        // Restart polling
        streamTimer = setInterval(streamStep, 1000);
      }, 100);
    }
  })
  .catch(err => {
    console.error("Stream fetch error:", err);
    toggleLiveStream();
  });
}

function updateStreamUIStatus(isFault, name) {
  const banner = document.getElementById("verdict-banner");
  const icon = document.getElementById("verdict-icon");
  const text = document.getElementById("verdict-text");
  const cls = document.getElementById("verdict-class");
  
  if (isFault) {
    banner.className = "verdict-banner fault";
    icon.textContent = "🚨";
    text.textContent = "ANOMALY INJECTED!";
    cls.textContent = "Defect Mode: " + name;
  } else {
    banner.className = "verdict-banner normal";
    icon.textContent = "🟢";
    text.textContent = "LIVE STREAM NORMAL";
    cls.textContent = "Status: Healthy Vibration Profile";
  }
}

// ── Interactive Hotspots Tour ─────────────────────────────────────
let activeMachine = 1;

function changeMachine(dir) {
  const container1 = document.getElementById("rig-container-1");
  const container2 = document.getElementById("rig-container-2");
  const title = document.getElementById("deploy-rig-title");
  const desc = document.getElementById("deploy-rig-desc");
  
  activeMachine += dir;
  if (activeMachine > 2) activeMachine = 1;
  if (activeMachine < 1) activeMachine = 2;
  
  if (activeMachine === 1) {
    container1.classList.remove("hidden");
    container2.classList.add("hidden");
    title.textContent = "L&T Hydraulic Valve Testing Rig — Edge Node Tour";
    desc.textContent = "Interactive schematic of a valve pressure test bench showing where microcontrollers (Edge Nodes) are deployed.";
  } else {
    container1.classList.add("hidden");
    container2.classList.remove("hidden");
    title.textContent = "Industrial Centrifugal Blower System — Edge Node Tour";
    desc.textContent = "Interactive schematic of a high-speed industrial blower fan showing sensor and edge node mounting locations.";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // Machine 1 Nodes
  const nodes1 = ['motor', 'pump', 'valve', 'plc'];
  nodes1.forEach(node => {
    const el = document.getElementById(`node-${node}`);
    const tooltip = document.getElementById(`tooltip-${node}`);
    if (el && tooltip) {
      el.addEventListener('mouseenter', () => tooltip.classList.remove('hidden'));
      el.addEventListener('mouseleave', () => tooltip.classList.add('hidden'));
    }
  });
  
  // Machine 2 Nodes
  const nodes2 = ['fanblade', 'fanbearing', 'fanflow', 'fanpanel'];
  nodes2.forEach(node => {
    const el = document.getElementById(`node-${node}`);
    const tooltip = document.getElementById(`tooltip-${node}`);
    if (el && tooltip) {
      el.addEventListener('mouseenter', () => tooltip.classList.remove('hidden'));
      el.addEventListener('mouseleave', () => tooltip.classList.add('hidden'));
    }
  });
});
