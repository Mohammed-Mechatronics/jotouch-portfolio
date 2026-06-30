/** UI controller for the JoTouch collection web app.
 *
 * Four-screen flow: Setup → Tests → Collect → Summary
 *
 * Tests are interactive: each test shows an instruction card, waits for
 * the operator to press "Ready", runs a 3-2-1 countdown, then samples.
 * After tests, a "Begin Collection" gate lets the operator position the
 * subject before recording starts.
 */

// ── WebSocket URLs ───────────────────────────────────────────────────────────

const sessionWsUrl = () => {
    const loc = window.location;
    const proto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${loc.host}/ws/session`;
};

const cameraWsUrl = () => {
    const loc = window.location;
    const proto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${loc.host}/ws/camera`;
};

// ── DOM references ───────────────────────────────────────────────────────────

const els = {
    // Setup screen
    sub: document.getElementById('sub'),
    ses: document.getElementById('ses'),
    port: document.getElementById('port'),
    nSensors: document.getElementById('n-sensors'),
    dryRun: document.getElementById('dry-run'),
    skipPrecollect: document.getElementById('skip-precollect'),
    nReps: document.getElementById('n-reps'),
    recordDuration: document.getElementById('record-duration'),
    prepDuration: document.getElementById('prep-duration'),
    restDuration: document.getElementById('rest-duration'),
    includeFreeform: document.getElementById('include-freeform'),
    startBtn: document.getElementById('start-btn'),
    cameraFeed: document.getElementById('camera-feed'),
    dryRunHint: document.getElementById('dry-run-hint'),
    setupFsrBars: document.getElementById('setup-fsr-bars'),
    setupFsrLabel: document.getElementById('setup-fsr-label'),
    cameraIndex: document.getElementById('camera-index'),
    scanCamerasBtn: document.getElementById('scan-cameras-btn'),
    cameraStatus: document.getElementById('camera-status'),
    scanPortsBtn: document.getElementById('scan-ports-btn'),
    fsrStatus: document.getElementById('fsr-status'),
    calibrateLedRoiBtn: document.getElementById('calibrate-led-roi-btn'),
    ledRoiStatus: document.getElementById('led-roi-status'),

    // Camera tracking settings
    camResolution: document.getElementById('cam-resolution'),
    camAutoExposure: document.getElementById('cam-auto-exposure'),
    camDetectionConf: document.getElementById('cam-detection-conf'),
    camDetectionConfVal: document.getElementById('cam-detection-conf-val'),
    camPresenceConf: document.getElementById('cam-presence-conf'),
    camPresenceConfVal: document.getElementById('cam-presence-conf-val'),
    camTrackingConf: document.getElementById('cam-tracking-conf'),
    camTrackingConfVal: document.getElementById('cam-tracking-conf-val'),
    applyTrackingBtn: document.getElementById('apply-tracking-btn'),
    trackingHint: document.getElementById('tracking-hint'),

    // Subject/session auto-suggest
    subjectModeRadios: document.querySelectorAll('input[name="subject-mode"]'),
    newSubjectRow: document.getElementById('new-subject-row'),
    existingSubjectRow: document.getElementById('existing-subject-row'),
    existingSubjectSelect: document.getElementById('existing-subject-select'),
    suggestSubjectBtn: document.getElementById('suggest-subject-btn'),
    suggestSessionBtn: document.getElementById('suggest-session-btn'),
    refreshSubjectsBtn: document.getElementById('refresh-subjects-btn'),

    // Mock banner
    mockBanner: document.getElementById('mock-banner'),
    mockBannerText: document.getElementById('mock-banner-text'),

    // Tests screen — instruction card
    testInstructionPanel: document.getElementById('test-instruction-panel'),
    testProgressLabel: document.getElementById('test-progress-label'),
    testInstructionLabel: document.getElementById('test-instruction-label'),
    testInstructionText: document.getElementById('test-instruction-text'),
    testCountdownDisplay: document.getElementById('test-countdown-display'),
    testFsrBars: document.getElementById('test-fsr-bars'),
    testReadyBtn: document.getElementById('test-ready-btn'),
    testCamera: document.getElementById('test-camera'),
    testCameraStatus: document.getElementById('test-camera-status'),
    testDurationHint: document.getElementById('test-duration-hint'),

    // Tests screen — results
    testList: document.getElementById('test-list'),
    overrideBtn: document.getElementById('override-btn'),
    retryBtn: document.getElementById('retry-btn'),
    backToSetupBtn: document.getElementById('back-to-setup-btn'),

    // Collect screen — begin gate
    beginGatePanel: document.getElementById('begin-gate-panel'),
    beginGateSummary: document.getElementById('begin-gate-summary'),
    beginCollectionBtn: document.getElementById('begin-collection-btn'),
    collectActive: document.getElementById('collect-active'),

    // Collect screen — active
    collectCamera: document.getElementById('collect-camera'),
    fsrBars: document.getElementById('fsr-bars'),
    cuePhase: document.getElementById('cue-phase'),
    cueTask: document.getElementById('cue-task'),
    cueDesc: document.getElementById('cue-desc'),
    cueInstruction: document.getElementById('cue-instruction'),
    cueCountdown: document.getElementById('cue-countdown'),
    quality: document.getElementById('quality'),
    progressFill: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),
    stopBtn: document.getElementById('stop-btn'),
    runList: document.getElementById('run-list'),
    collectBackToSetupBtn: document.getElementById('collect-back-to-setup-btn'),

    // Summary screen
    summaryContent: document.getElementById('summary-content'),
    qualityReport: document.getElementById('quality-report'),
    log: document.getElementById('log'),
    restartBtn: document.getElementById('restart-btn'),

    // Shared
    connectionStatus: document.getElementById('connection-status'),
    steps: {
        setup: document.getElementById('step-setup'),
        tests: document.getElementById('step-tests'),
        collect: document.getElementById('step-collect'),
        summary: document.getElementById('step-summary'),
    },
    screens: {
        setup: document.getElementById('screen-setup'),
        tests: document.getElementById('screen-tests'),
        collect: document.getElementById('screen-collect'),
        summary: document.getElementById('screen-summary'),
    },
};

// ── State ────────────────────────────────────────────────────────────────────

let sessionWs = null;
let cameraWs = null;
let sessionReconnectDelay = 1500;
let cameraReconnectDelay = 2000;
let cameraUrl = null;
let totalRuns = 0;
let completedRuns = 0;
let testResults = [];
let qualityHistory = [];
let runHistory = [];
let sessionConfig = {};
let mockMode = { sensor: false, camera: false };
let currentTestName = '';
let currentTestLabel = '';
let testsStillRunning = false; // True while precollect tests are in progress
let navigatingBack = false;  // Prevents auto-advance when returning to Setup

// ── Screen management ────────────────────────────────────────────────────────

let currentScreen = 'setup';

function showScreen(name) {
    currentScreen = name;
    for (const [key, el] of Object.entries(els.screens)) {
        el.classList.toggle('active', key === name);
    }
    for (const [key, el] of Object.entries(els.steps)) {
        el.classList.remove('active');
        el.classList.remove('done');
    }
    const order = ['setup', 'tests', 'collect', 'summary'];
    const idx = order.indexOf(name);
    for (let i = 0; i < idx; i++) {
        els.steps[order[i]].classList.add('done');
    }
    els.steps[name].classList.add('active');

    // FSR polling: only on Setup screen (Tests + Collect use WebSocket events)
    if (name === 'setup') {
        startFsrPolling();
    } else {
        stopFsrPolling();
    }
}

function advanceToCollect() {
    if (currentScreen === 'tests' || currentScreen === 'setup') {
        showScreen('collect');
    }
}

// ── Utilities ────────────────────────────────────────────────────────────────

function log(msg) {
    const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
    els.log.textContent += line + '\n';
    els.log.scrollTop = els.log.scrollHeight;
}

function setConnectionStatus(connected) {
    const badge = els.connectionStatus;
    badge.textContent = connected ? 'Connected' : 'Disconnected';
    badge.className = `badge ${connected ? 'connected' : 'disconnected'}`;
}

function showErrorBanner(msg) {
    let banner = document.getElementById('error-banner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'error-banner';
        banner.className = 'error-banner';
        const screen = document.getElementById('screen-setup');
        screen.insertBefore(banner, screen.firstChild);
    }
    banner.innerHTML = `<strong>Error:</strong> ${msg} <button onclick="this.parentElement.remove()">\u2715</button>`;
    banner.style.display = 'block';
}

function setQuality(level, reason, per_sensor = null) {
    els.quality.className = `quality ${level}`;
    els.quality.textContent = reason || level;
    qualityHistory.push({ level, reason, per_sensor, time: new Date().toISOString() });
}

function updateProgress() {
    const pct = totalRuns > 0 ? (completedRuns / totalRuns) * 100 : 0;
    els.progressFill.style.width = `${pct}%`;
    els.progressText.textContent = `${completedRuns} / ${totalRuns}`;
}

// ── Mock banner ──────────────────────────────────────────────────────────────

function updateMockBanner() {
    const parts = [];
    if (mockMode.sensor) parts.push('sensor');
    if (mockMode.camera) parts.push('camera');
    if (parts.length > 0) {
        els.mockBannerText.textContent = `Using simulated ${parts.join(' + ')} data`;
        els.mockBanner.classList.add('active');
    } else {
        els.mockBanner.classList.remove('active');
    }
}

// ── FSR bars ─────────────────────────────────────────────────────────────────

function createFsrBars(container, prefix) {
    container.innerHTML = '';
    for (let i = 0; i < 4; i++) {
        const row = document.createElement('div');
        row.className = 'fsr-bar';
        row.innerHTML = `
            <div class="fsr-label">FSR${i}</div>
            <div class="fsr-track"><div class="fsr-fill" id="${prefix}-fsr${i}-fill"></div></div>
            <div class="fsr-value" id="${prefix}-fsr${i}-value">0</div>
        `;
        container.appendChild(row);
    }
}

function updateFsr(values) {
    if (!values || values.length === 0) return;
    // Update all three FSR bar sets: setup, test, collect
    for (let i = 0; i < values.length; i++) {
        const v = values[i];
        const pct = Math.min(100, Math.max(0, (v / 1023) * 100));
        for (const prefix of ['setup', 'test', 'collect']) {
            const fill = document.getElementById(`${prefix}-fsr${i}-fill`);
            const value = document.getElementById(`${prefix}-fsr${i}-value`);
            if (fill) {
                if (prefix === 'setup') {
                    // Vertical bars: fill from bottom
                    fill.style.height = `${pct}%`;
                    fill.style.width = '100%';
                } else {
                    // Horizontal bars: fill from left
                    fill.style.width = `${pct}%`;
                }
            }
            if (value) value.textContent = v;
        }
    }
}

// ── FSR polling (Setup screen only) ───────────────────────────────────────────

let fsrPollTimer = null;
let fsrPollPending = false;

async function pollFsr() {
    if (fsrPollPending) return; // skip if previous request still in flight
    fsrPollPending = true;
    try {
        // Pass the current port/dry_run/n_sensors from the Setup form
        // so the backend opens a real serial reader for live preview
        const params = new URLSearchParams();
        const portVal = els.port ? els.port.value : '';
        if (portVal) params.set('port', portVal);
        if (els.nSensors) params.set('n_sensors', els.nSensors.value || 4);
        const isDryRun = els.dryRun ? els.dryRun.checked : false;
        params.set('dry_run', isDryRun ? 'true' : 'false');
        const resp = await fetch('/api/fsr?' + params.toString());
        const data = await resp.json();
        if (data.values && data.values.length > 0) {
            updateFsr(data.values);
        }

        // Update source label and status
        const label = els.setupFsrLabel;
        const status = els.fsrStatus;
        if (data.source === 'mock') {
            if (label) label.textContent = 'FSR Preview (MOCK DATA)';
            if (status) status.textContent = 'Using simulated data. Select a port and uncheck Dry run for live data.';
        } else if (data.source === 'serial') {
            if (data.available) {
                if (label) label.textContent = 'FSR Preview (LIVE)';
                if (status) status.textContent = `Connected to ${portVal}. Reading live sensor data.`;
            } else {
                if (label) label.textContent = 'FSR Preview (NO DATA)';
                if (status) status.textContent = `Port ${portVal} opened but no data received. Check Arduino is powered and sending data.`;
            }
        } else if (data.source === 'session') {
            if (label) label.textContent = 'FSR Preview (Session Active)';
            if (status) status.textContent = 'Session is running. Data from active session.';
        } else if (data.source === 'none') {
            if (label) label.textContent = 'FSR Preview';
            if (status) status.textContent = isDryRun
                ? 'Dry run is ON — using simulated data.'
                : (portVal ? `Cannot open ${portVal}. Check port name and close other apps.` : 'No port selected. Select a port or enable Dry run.');
        } else if (data.source === 'error') {
            if (label) label.textContent = 'FSR Preview (ERROR)';
            if (status) status.textContent = 'Error: ' + (data.error || 'Unknown');
        }
    } catch (e) {
        // Server not reachable — silently ignore
    } finally {
        fsrPollPending = false;
    }
}

function scheduleFsrPoll() {
    if (!fsrPollTimer) return;
    fsrPollTimer = setTimeout(async () => {
        await pollFsr();
        scheduleFsrPoll();
    }, 50); // 20 Hz target, but only after previous request finishes
}

function startFsrPolling() {
    if (fsrPollTimer) return;
    fsrPollTimer = true; // used as a flag that polling is active
    scheduleFsrPoll();
    // Also run one poll immediately so the bars appear right away
    setTimeout(pollFsr, 0);
}

function stopFsrPolling() {
    if (fsrPollTimer) {
        clearTimeout(fsrPollTimer);
        fsrPollTimer = null;
    }
    fsrPollPending = false;
}

// ── Serial port scanning ─────────────────────────────────────────────────────

async function scanPorts() {
    if (els.scanPortsBtn) els.scanPortsBtn.disabled = true;
    try {
        const resp = await fetch('/api/ports');
        const data = await resp.json();
        if (data.ports && els.port) {
            els.port.innerHTML = '<option value="">-- Select port --</option>';
            for (const p of data.ports) {
                const opt = document.createElement('option');
                opt.value = p.device;
                opt.textContent = `${p.device} — ${p.description}`;
                els.port.appendChild(opt);
            }
            // Auto-select USB serial devices (likely Arduino)
            const usb = data.ports.find(p => p.description.includes('USB') || p.manufacturer.includes('Arduino'));
            if (usb) els.port.value = usb.device;
        }
    } catch (e) {
        console.error('Failed to scan ports:', e);
    } finally {
        if (els.scanPortsBtn) els.scanPortsBtn.disabled = false;
    }
}

// ── LED ROI calibration ──────────────────────────────────────────────────────

async function calibrateLedRoi() {
    if (els.calibrateLedRoiBtn) els.calibrateLedRoiBtn.disabled = true;
    if (els.ledRoiStatus) els.ledRoiStatus.textContent = 'Opening calibration window... (click+drag to select LED, s=save, q=quit)';
    try {
        const cameraIdx = els.cameraIndex ? parseInt(els.cameraIndex.value, 10) || 0 : 0;
        const resp = await fetch(`/api/led_roi/calibrate?camera_index=${cameraIdx}`, { method: 'POST' });
        const data = await resp.json();
        if (data.ok) {
            if (els.ledRoiStatus) {
                els.ledRoiStatus.textContent = `ROI saved: ${data.roi.width}x${data.roi.height} at (${data.roi.x}, ${data.roi.y})`;
                els.ledRoiStatus.style.color = 'var(--green)';
            }
            log(`LED ROI calibrated: ${JSON.stringify(data.roi)}`);
        } else {
            if (els.ledRoiStatus) {
                els.ledRoiStatus.textContent = 'Calibration failed: ' + (data.error || 'Unknown');
                els.ledRoiStatus.style.color = 'var(--warning)';
            }
        }
    } catch (e) {
        if (els.ledRoiStatus) {
            els.ledRoiStatus.textContent = 'Calibration error: ' + e.message;
            els.ledRoiStatus.style.color = 'var(--warning)';
        }
    } finally {
        if (els.calibrateLedRoiBtn) els.calibrateLedRoiBtn.disabled = false;
    }
}

async function loadLedRoiStatus() {
    if (!els.ledRoiStatus) return;
    try {
        const resp = await fetch('/api/led_roi');
        const data = await resp.json();
        if (data.ok && data.roi) {
            els.ledRoiStatus.textContent = `ROI loaded: ${data.roi.width}x${data.roi.height} at (${data.roi.x}, ${data.roi.y})`;
            els.ledRoiStatus.style.color = 'var(--green)';
        }
    } catch (e) {
        // Ignore — will show default "No ROI saved" message
    }
}

// ── Camera tracking settings ──────────────────────────────────────────────────

async function loadCameraSettings() {
    try {
        const resp = await fetch('/api/camera_settings');
        const data = await resp.json();
        if (data.ok && data.settings) {
            const s = data.settings;
            if (els.camResolution) els.camResolution.value = s.resolution || '640x480';
            if (els.camAutoExposure) els.camAutoExposure.value = String(s.auto_exposure ?? true);
            if (els.camDetectionConf) {
                els.camDetectionConf.value = s.min_detection_confidence ?? 0.5;
                els.camDetectionConfVal.textContent = parseFloat(els.camDetectionConf.value).toFixed(2);
            }
            if (els.camPresenceConf) {
                els.camPresenceConf.value = s.min_presence_confidence ?? 0.5;
                els.camPresenceConfVal.textContent = parseFloat(els.camPresenceConf.value).toFixed(2);
            }
            if (els.camTrackingConf) {
                els.camTrackingConf.value = s.min_tracking_confidence ?? 0.5;
                els.camTrackingConfVal.textContent = parseFloat(els.camTrackingConf.value).toFixed(2);
            }
        }
    } catch (e) {
        // Ignore — settings panel just shows defaults
    }
}

async function applyCameraSettings() {
    if (els.applyTrackingBtn) els.applyTrackingBtn.disabled = true;
    if (els.trackingHint) {
        els.trackingHint.textContent = 'Applying settings... (camera may restart)';
        els.trackingHint.style.color = 'var(--warning)';
    }
    try {
        const body = {
            resolution: els.camResolution ? els.camResolution.value : '640x480',
            auto_exposure: els.camAutoExposure ? els.camAutoExposure.value === 'true' : true,
            min_detection_confidence: els.camDetectionConf ? parseFloat(els.camDetectionConf.value) : 0.5,
            min_presence_confidence: els.camPresenceConf ? parseFloat(els.camPresenceConf.value) : 0.5,
            min_tracking_confidence: els.camTrackingConf ? parseFloat(els.camTrackingConf.value) : 0.5,
        };
        const resp = await fetch('/api/camera_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (data.ok) {
            const restartMsg = data.needs_restart ? ' (camera restarted)' : '';
            if (els.trackingHint) {
                els.trackingHint.textContent = `Settings applied: ${data.changed.join(', ')}${restartMsg}`;
                els.trackingHint.style.color = 'var(--green)';
            }
            log(`Camera settings updated: ${JSON.stringify(data)}`);
        } else {
            if (els.trackingHint) {
                els.trackingHint.textContent = 'Failed: ' + (data.error || 'Unknown');
                els.trackingHint.style.color = 'var(--warning)';
            }
        }
    } catch (e) {
        if (els.trackingHint) {
            els.trackingHint.textContent = 'Error: ' + e.message;
            els.trackingHint.style.color = 'var(--warning)';
        }
    } finally {
        if (els.applyTrackingBtn) els.applyTrackingBtn.disabled = false;
    }
}

// ── Cue card ─────────────────────────────────────────────────────────────────

function updateCue(event) {
    const phase = (event.phase || 'READY').toUpperCase();
    const task = event.task || '\u2014';
    const remaining = event.remaining_s !== undefined ? event.remaining_s.toFixed(1) + 's' : '';

    els.cuePhase.textContent = phase;
    // Show the human-readable display name if available, otherwise the raw task
    els.cueTask.textContent = event.display_name || task;
    // Show the movement description and patient instruction
    els.cueDesc.textContent = event.description || '';
    els.cueInstruction.textContent = event.instruction || '';
    els.cueCountdown.textContent = remaining || '--';

    const card = document.getElementById('cue-card');
    card.style.borderColor = phase === 'RECORD' ? 'var(--green)' :
                              phase === 'PREP' ? 'var(--yellow)' :
                              phase === 'REST' ? 'var(--muted)' : 'transparent';
}

// ── Run list ─────────────────────────────────────────────────────────────────

function addRunToLog(event) {
    runHistory.push(event);
    const item = document.createElement('div');
    item.className = 'run-item';
    item.innerHTML = `
        <div class="run-status pass">\u2713</div>
        <div class="run-name">${event.task} (run ${event.run})</div>
        <div class="run-detail">${event.physio_samples} phys / ${event.camera_frames} cam</div>
    `;
    els.runList.appendChild(item);
    els.runList.scrollTop = els.runList.scrollHeight;
}

function addSkippedRunToLog(event) {
    runHistory.push(event);
    const item = document.createElement('div');
    item.className = 'run-item';
    const reasonText = event.reason === 'stopped' ? 'stopped' :
                       event.reason === 'interrupted' ? 'interrupted' :
                       event.reason === 'crashed' ? 'crashed' : 'skipped';
    item.innerHTML = `
        <div class="run-status skip">\u2717</div>
        <div class="run-name">${event.task} (run ${event.run})</div>
        <div class="run-detail">${reasonText} \u2014 quarantined</div>
    `;
    els.runList.appendChild(item);
    els.runList.scrollTop = els.runList.scrollHeight;
}

// ── Tests screen ─────────────────────────────────────────────────────────────

function setRetryLinksEnabled(enabled) {
    // Enable or disable all per-test Retry buttons in the results list.
    // Disabled while tests are still running so the user can't trigger a
    // retry that would interfere with the current test sequence.
    for (const btn of els.testList.querySelectorAll('.retry-link')) {
        btn.disabled = !enabled;
        btn.title = enabled ? '' : 'Tests still running — retry available when all tests finish';
    }
}

function addTestResult(event) {
    // Replace any existing result item with the same test name (e.g., after retry)
    const existingIndex = testResults.findIndex(t => t.name === event.name);
    if (existingIndex >= 0) {
        testResults[existingIndex] = event;
        // Remove the existing DOM item; the new one will be appended at the end
        const existingItem = els.testList.children[existingIndex];
        if (existingItem) {
            existingItem.remove();
        }
    } else {
        testResults.push(event);
    }

    const item = document.createElement('div');
    item.className = 'test-item';
    const icon = event.passed ? '\u2713' : '\u2717';
    const cls = event.passed ? 'pass' : 'fail';
    // Use the human-readable label if we have it, otherwise the raw name
    const label = currentTestLabel || event.name;
    // Add a retry link for failed tests.
    // Disabled while tests are still running to prevent mid-run interference.
    let retryHtml = '';
    if (!event.passed) {
        const disabled = testsStillRunning ? 'disabled title="Tests still running"' : '';
        retryHtml = `<button class="retry-link" data-test-name="${event.name}" ${disabled}>Retry</button>`;
    }

    // Build collapsible details section from the per-channel metrics
    let detailsHtml = '';
    if (event.details && Object.keys(event.details).length > 0) {
        const detailsJson = JSON.stringify(event.details);
        detailsHtml = `<details class="test-details"><summary>Details</summary><pre>${escapeHtml(detailsJson)}</pre></details>`;
    }

    item.innerHTML = `
        <div class="test-icon ${cls}">${icon}</div>
        <div class="test-name">${label}</div>
        <div class="test-msg">${event.message || ''}${retryHtml}</div>
        ${detailsHtml}
    `;
    // Wire up the retry link if present
    const retryBtn = item.querySelector('.retry-link');
    if (retryBtn) {
        retryBtn.addEventListener('click', () => {
            const testName = retryBtn.getAttribute('data-test-name');
            sendCommand('retry_test', { name: testName });
            log(`Retrying test: ${testName}`);
        });
    }
    els.testList.appendChild(item);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function clearTestList() {
    els.testList.innerHTML = '';
    testResults = [];
    els.testInstructionLabel.textContent = 'Preparing\u2026';
    els.testInstructionText.textContent = '';
    els.testCountdownDisplay.textContent = '';
    els.testCountdownDisplay.className = 'test-countdown-display';
    els.testReadyBtn.style.display = 'none';
}

// ── Begin Collection gate ────────────────────────────────────────────────────

function renderBeginGate(event) {
    let html = `<div class="summary-row"><span class="label">Tests passed</span><span class="value">${event.tests_passed} / ${event.tests_total}</span></div>`;
    html += `<div class="summary-row"><span class="label">Total runs</span><span class="value">${event.total_runs}</span></div>`;
    if (event.run_summary && event.run_summary.length > 0) {
        html += '<div style="margin-top:0.5rem;"><strong>Run plan:</strong></div>';
        for (const r of event.run_summary) {
            html += `<div class="begin-gate-run"><span class="run-task">${r.task} (run ${r.run}, rep ${r.rep})</span><span class="run-dur">${r.duration_s}s</span></div>`;
        }
    }
    els.beginGateSummary.innerHTML = html;
    els.beginGatePanel.style.display = 'block';
    els.collectActive.style.display = 'none';
}

// ── Summary screen ───────────────────────────────────────────────────────────

function renderSummary(event) {
    const allPassed = testResults.length > 0 && testResults.every(t => t.passed);
    const warnings = qualityHistory.filter(q => q.level === 'yellow' || q.level === 'red');

    let html = '';
    html += `<div class="summary-row"><span class="label">Subject</span><span class="value">${sessionConfig.sub || '\u2014'}</span></div>`;
    html += `<div class="summary-row"><span class="label">Session</span><span class="value">${sessionConfig.ses || '\u2014'}</span></div>`;
    html += `<div class="summary-row"><span class="label">Mode</span><span class="value">${sessionConfig.dry_run ? 'Dry run' : 'Live'}</span></div>`;
    html += `<div class="summary-row"><span class="label">Runs completed</span><span class="value">${event.completed_runs} / ${event.total_runs}</span></div>`;
    if (event.quarantined_runs && event.quarantined_runs > 0) {
        html += `<div class="summary-row"><span class="label">Runs quarantined</span><span class="value" style="color:var(--warning);">${event.quarantined_runs} (aborted/partial \u2014 moved to _partial/)</span></div>`;
    }
    html += `<div class="summary-row"><span class="label">Physio samples</span><span class="value">${event.total_physio_samples}</span></div>`;
    html += `<div class="summary-row"><span class="label">Camera frames</span><span class="value">${event.total_camera_frames}</span></div>`;
    if (event.session_dir) {
        html += `<div class="summary-row"><span class="label">BIDS path</span><span class="value">${event.session_dir}</span></div>`;
    }
    els.summaryContent.innerHTML = html;

    let qhtml = '';
    if (testResults.length > 0) {
        qhtml += `<div class="quality-flag"><span class="flag-level ${allPassed ? 'green' : 'red'}">${allPassed ? '\u2713' : '\u2717'}</span><span>Pre-collection tests: ${testResults.filter(t => t.passed).length}/${testResults.length} passed</span></div>`;
    }
    if (warnings.length > 0) {
        const reds = warnings.filter(w => w.level === 'red');
        const yellows = warnings.filter(w => w.level === 'yellow');
        if (reds.length > 0) {
            qhtml += `<div class="quality-flag"><span class="flag-level red">!</span><span>${reds.length} red quality alert(s) during collection</span></div>`;
        }
        if (yellows.length > 0) {
            qhtml += `<div class="quality-flag"><span class="flag-level yellow">~</span><span>${yellows.length} yellow quality warning(s) during collection</span></div>`;
        }
    }
    if (!qhtml) {
        qhtml = '<div class="quality-flag"><span class="flag-level green">\u2713</span><span>No quality issues detected</span></div>';
    }

    // Per-sensor breakdown from the latest quality event with per-sensor data
    const latestWithPerSensor = qualityHistory.slice().reverse().find(q => q.per_sensor);
    if (latestWithPerSensor && latestWithPerSensor.per_sensor && latestWithPerSensor.per_sensor.flat_pct && latestWithPerSensor.per_sensor.zero_pct) {
        const flat = latestWithPerSensor.per_sensor.flat_pct;
        const zero = latestWithPerSensor.per_sensor.zero_pct;
        let sensorHtml = '<div class="per-sensor-quality"><strong>Latest per-sensor quality:</strong><table>';
        sensorHtml += '<tr><th>Sensor</th><th>Flat %</th><th>Zero %</th></tr>';
        for (let i = 0; i < flat.length; i++) {
            sensorHtml += `<tr><td>FSR ${i}</td><td>${flat[i].toFixed(1)}%</td><td>${zero[i].toFixed(1)}%</td></tr>`;
        }
        sensorHtml += '</table></div>';
        qhtml += sensorHtml;
    }

    els.qualityReport.innerHTML = qhtml;
}

// ── Event handler ────────────────────────────────────────────────────────────

function handleEvent(event) {
    // While navigating back to Setup, ignore all session events except the
    // session_ended handshake and errors. This prevents stray setup/state/fsr
    // events from auto-advancing the screen before the backend has fully stopped.
    if (navigatingBack && event.type !== 'session_ended' && event.type !== 'error') {
        log(`Ignored ${event.type} event while navigating back`);
        return;
    }

    switch (event.type) {
        case 'session_ended':
            if (navigatingBack) {
                navigatingBack = false;
                log('Session ended — returned to Setup');
            } else {
                log('Session ended');
            }
            // Re-enable the Start button in case it was left disabled
            els.startBtn.disabled = false;
            els.stopBtn.disabled = true;
            break;
        case 'setup':
            // Ignore setup events if we're navigating back to Setup
            if (navigatingBack) {
                log('Ignored setup event (navigating back to Setup)');
                break;
            }
            log(`Session started: ${event.sub} / ${event.ses} (${event.dry_run ? 'dry run' : 'live'})`);
            totalRuns = event.total_runs;
            completedRuns = 0;
            updateProgress();
            setQuality('green', 'Running');
            testsStillRunning = !els.skipPrecollect.checked;
            if (!els.skipPrecollect.checked) {
                showScreen('tests');
            } else {
                showScreen('collect');
            }
            break;
        case 'mock_mode':
            mockMode.sensor = event.sensor_mock;
            mockMode.camera = event.camera_mock;
            updateMockBanner();
            log(`Mock mode: sensor=${event.sensor_mock}, camera=${event.camera_mock}`);
            break;
        case 'test_instruction':
            currentTestName = event.name;
            currentTestLabel = event.label || event.name;
            els.testProgressLabel.textContent = `Test ${event.test_index + 1} of ${event.total_tests}`;
            els.testInstructionLabel.textContent = event.label || event.name;
            els.testInstructionText.textContent = event.instruction || '';
            if (els.testDurationHint && event.duration_s) {
                els.testDurationHint.textContent = `Duration: ~${event.duration_s}s`;
            } else if (els.testDurationHint) {
                els.testDurationHint.textContent = '';
            }
            els.testCountdownDisplay.textContent = '';
            els.testCountdownDisplay.className = 'test-countdown-display';
            els.testReadyBtn.style.display = 'inline-block';
            els.testReadyBtn.textContent = 'Ready';
            if (els.retryBtn) els.retryBtn.style.display = 'none';
            log(`Test instruction: ${event.label}`);
            break;
        case 'test_ready':
            els.testCountdownDisplay.textContent = 'Press Ready';
            els.testCountdownDisplay.className = 'test-countdown-display ready';
            els.testReadyBtn.style.display = 'inline-block';
            break;
        case 'test_countdown':
            els.testReadyBtn.style.display = 'none';
            els.testCountdownDisplay.textContent = String(event.countdown);
            els.testCountdownDisplay.className = 'test-countdown-display';
            break;
        case 'test_running':
            els.testCountdownDisplay.textContent = 'GO';
            els.testCountdownDisplay.className = 'test-countdown-display go';
            break;
        case 'test':
            addTestResult(event);
            log(`Test ${event.passed ? 'PASS' : 'FAIL'}: ${currentTestLabel || event.name} \u2014 ${event.message}`);
            break;
        case 'retry_started':
            // Backend is re-running failed tests — go back to the tests screen
            // so the instruction/countdown/result events are visible.
            testsStillRunning = true;
            setRetryLinksEnabled(false);
            showScreen('tests');
            log(`Retrying test(s): ${(event.tests || []).join(', ')}`);
            break;
        case 'collection_ready':
            // All tests done — enable retry-links now that tests aren't running.
            testsStillRunning = false;
            setRetryLinksEnabled(true);
            log(`Collection ready: ${event.tests_passed}/${event.tests_total} tests passed, ${event.total_runs} runs planned`);
            showScreen('collect');
            renderBeginGate(event);
            // Show retry button if any tests failed
            if (els.retryBtn) {
                els.retryBtn.style.display = (event.tests_passed < event.tests_total) ? 'inline-block' : 'none';
            }
            break;
        case 'state':
            // Collection has begun — show active collect UI
            if (els.beginGatePanel.style.display !== 'none') {
                els.beginGatePanel.style.display = 'none';
                els.collectActive.style.display = 'block';
            }
            advanceToCollect();
            updateCue(event);
            break;
        case 'fsr':
            updateFsr(event.values);
            break;
        case 'camera':
            // Camera metadata: hand detection confidence
            if (els.cameraStatus && currentScreen === 'collect') {
                if (event.valid) {
                    els.cameraStatus.textContent = `Hand: ${(event.confidence * 100).toFixed(0)}% confidence`;
                } else {
                    els.cameraStatus.textContent = 'No hand detected';
                }
            }
            break;
        case 'quality':
            setQuality(event.level, event.reason, event.per_sensor);
            break;
        case 'run_complete':
            advanceToCollect();
            completedRuns++;
            updateProgress();
            addRunToLog(event);
            log(`Run complete: ${event.task} run ${event.run} \u2014 ${event.physio_samples} physio, ${event.camera_frames} camera`);
            break;
        case 'run_skipped':
            addSkippedRunToLog(event);
            log(`Run skipped: ${event.task} run ${event.run} \u2014 ${event.reason}, quarantined`);
            break;
        case 'summary':
            log(`Session complete: ${event.completed_runs}/${event.total_runs} runs, ${event.total_physio_samples} physio samples`);
            setQuality('green', 'Done');
            renderSummary(event);
            showScreen('summary');
            stopCamera();
            els.stopBtn.disabled = true;
            break;
        case 'warning':
            log(`Warning: ${event.message}`);
            setQuality('yellow', event.message);
            break;
        case 'error':
            log(`Error: ${event.message}`);
            setQuality('red', event.message);
            if (currentScreen === 'setup' || currentScreen === 'tests') {
                els.startBtn.disabled = false;
                showErrorBanner(event.message);
            }
            break;
        default:
            break;
    }
}

// ── WebSocket management ──────────────────────────────────────────────────────

function connectSession() {
    if (sessionWs) return;

    sessionWs = new WebSocket(sessionWsUrl());
    sessionWs.onopen = () => {
        setConnectionStatus(true);
        sessionReconnectDelay = 1500; // reset backoff on successful connect
        log('Session WebSocket connected');
    };
    sessionWs.onmessage = (e) => {
        const event = JSON.parse(e.data);
        handleEvent(event);
    };
    sessionWs.onclose = () => {
        setConnectionStatus(false);
        log('Session WebSocket disconnected');
        sessionWs = null;
        if (currentScreen === 'setup' || currentScreen === 'tests') {
            els.startBtn.disabled = false;
        }
        // Do not auto-reconnect while navigating back to Setup (the next
        // Start Session will reconnect explicitly) or on the Summary screen.
        if (!navigatingBack && currentScreen !== 'summary') {
            log(`Reconnecting session WebSocket in ${sessionReconnectDelay}ms...`);
            setTimeout(connectSession, sessionReconnectDelay);
            sessionReconnectDelay = Math.min(sessionReconnectDelay * 2, 30000);
        }
    };
    sessionWs.onerror = (e) => {
        log('Session WebSocket error');
        console.error(e);
    };
}

function connectCamera() {
    if (cameraWs) return;

    cameraWs = new WebSocket(cameraWsUrl());
    // WS-4: Use arraybuffer to parse the 1-byte type prefix (0x00=JPEG, 0x01=JSON)
    cameraWs.binaryType = 'arraybuffer';
    cameraWs.onopen = () => {
        cameraReconnectDelay = 2000; // reset backoff on successful connect
    };
    cameraWs.onmessage = (e) => {
        if (e.data instanceof ArrayBuffer) {
            const data = new Uint8Array(e.data);
            if (data.length < 2) return;
            const prefix = data[0];
            if (prefix === 0x00) {
                // JPEG frame — create blob from the rest of the buffer
                if (cameraUrl) URL.revokeObjectURL(cameraUrl);
                const jpegBlob = new Blob([data.slice(1)], { type: 'image/jpeg' });
                cameraUrl = URL.createObjectURL(jpegBlob);
                els.cameraFeed.src = cameraUrl;
                els.collectCamera.src = cameraUrl;
                if (els.testCamera) els.testCamera.src = cameraUrl;
            } else if (prefix === 0x01) {
                // JSON metadata frame — parse and update overlay
                try {
                    const jsonStr = new TextDecoder().decode(data.slice(1));
                    const meta = JSON.parse(jsonStr);
                    updateCameraOverlay(meta);
                } catch (err) {
                    console.error('Camera metadata parse error', err);
                }
            }
        } else if (e.data instanceof Blob) {
            // Backward compat: old-style raw JPEG blob (no prefix)
            if (cameraUrl) URL.revokeObjectURL(cameraUrl);
            cameraUrl = URL.createObjectURL(e.data);
            els.cameraFeed.src = cameraUrl;
            els.collectCamera.src = cameraUrl;
            if (els.testCamera) els.testCamera.src = cameraUrl;
        }
    };
    cameraWs.onclose = () => {
        cameraWs = null;
        // Auto-reconnect with exponential backoff, unless we are on the summary screen.
        if (currentScreen !== 'summary') {
            setTimeout(connectCamera, cameraReconnectDelay);
            cameraReconnectDelay = Math.min(cameraReconnectDelay * 2, 30000);
        }
    };
    cameraWs.onerror = (e) => {
        console.error('Camera WebSocket error', e);
    };
}

// WS-4: Update the camera overlay with metadata from the WS
function updateCameraOverlay(meta) {
    // Update every camera overlay (Setup, precollect test-camera, collect-camera)
    // from a single metadata event so all three panels show fps/LED/hand.
    const overlayIds = ['camera-overlay', 'test-camera-overlay', 'collect-camera-overlay'];
    for (const overlayId of overlayIds) {
        const overlay = document.getElementById(overlayId);
        if (!overlay) continue;
        const fpsEl = overlay.querySelector('.overlay-fps');
        const ledEl = overlay.querySelector('.overlay-led');
        const handEl = overlay.querySelector('.overlay-hand');
        const roiEl = overlay.querySelector('.overlay-roi');
        if (fpsEl) fpsEl.textContent = meta.fps != null ? meta.fps.toFixed(1) + ' fps' : '-- fps';
        if (ledEl) {
            const brightness = meta.led_brightness;
            if (brightness != null) {
                ledEl.textContent = 'LED: ' + brightness.toFixed(0);
                ledEl.className = brightness > 50 ? 'overlay-led on' : 'overlay-led off';
            } else {
                ledEl.textContent = 'LED: --';
                ledEl.className = 'overlay-led';
            }
        }
        if (handEl) {
            if (meta.valid) {
                handEl.textContent = 'Hand: ' + (meta.confidence * 100).toFixed(0) + '%';
                handEl.className = 'overlay-hand valid';
            } else {
                handEl.textContent = 'Hand: none';
                handEl.className = 'overlay-hand';
            }
        }
        if (roiEl) {
            roiEl.textContent = meta.roi ? 'ROI: set' : 'ROI: none';
            roiEl.className = meta.roi ? 'overlay-roi set' : 'overlay-roi';
        }
    }
}

// ── Camera scanning ──────────────────────────────────────────────────────────

async function scanCameras() {
    if (els.scanCamerasBtn) els.scanCamerasBtn.disabled = true;
    if (els.cameraStatus) els.cameraStatus.textContent = 'Scanning...';
    try {
        const resp = await fetch('/api/cameras');
        const data = await resp.json();
        if (data.cameras) {
            const available = data.cameras.filter(c => c.available);
            if (els.cameraStatus) {
                els.cameraStatus.textContent = available.length > 0
                    ? `${available.length} camera(s) found: index ${available.map(c => c.index).join(', ')}`
                    : 'No cameras found';
            }
            // Populate dropdown
            if (els.cameraIndex) {
                els.cameraIndex.innerHTML = '';
                for (const cam of data.cameras) {
                    const opt = document.createElement('option');
                    opt.value = cam.index;
                    opt.textContent = `Camera ${cam.index}${cam.available ? '' : ' (not available)'}`;
                    if (!cam.available) opt.disabled = true;
                    els.cameraIndex.appendChild(opt);
                }
                // Select first available
                const firstAvail = data.cameras.find(c => c.available);
                if (firstAvail) els.cameraIndex.value = firstAvail.index;
            }
        }
    } catch (e) {
        if (els.cameraStatus) els.cameraStatus.textContent = 'Scan failed: ' + e.message;
    } finally {
        if (els.scanCamerasBtn) els.scanCamerasBtn.disabled = false;
    }
}

function stopCamera() {
    if (cameraWs) {
        cameraWs.close();
        cameraWs = null;
    }
    if (cameraUrl) {
        URL.revokeObjectURL(cameraUrl);
        cameraUrl = null;
    }
}

// ── Session control ──────────────────────────────────────────────────────────

function sendCommand(cmd, payload = null) {
    if (sessionWs && sessionWs.readyState === WebSocket.OPEN) {
        const message = { command: cmd };
        if (payload) {
            Object.assign(message, payload);
        }
        sessionWs.send(JSON.stringify(message));
        return true;
    }
    log(`Cannot send ${cmd} — WebSocket not ready`);
    return false;
}

function startSession() {
    // Subject/session IDs are auto-generated and readonly. We still validate
    // that the auto-suggested value is a legal BIDS label, and surface a
    // clear message if auto-suggest left the field empty.
    const subVal = els.sub.value.trim();
    const sesVal = els.ses.value.trim();
    if (!subVal || !/^[A-Za-z0-9]+$/.test(subVal)) {
        log('ERROR: Could not determine a valid Subject ID. Check the auto-suggest service or refresh.');
        return;
    }
    if (!sesVal || !/^[A-Za-z0-9]+$/.test(sesVal)) {
        log('ERROR: Could not determine a valid Session ID. Check the auto-suggest service or refresh.');
        return;
    }

    sessionConfig = {
        sub: subVal,
        ses: sesVal,
        dry_run: els.dryRun.checked,
    };

    // Stop FSR polling immediately — the session needs exclusive access
    // to the serial port. If polling continues, it will try to reopen
    // the port and conflict with the session's SerialSensorReader.
    stopFsrPolling();

    clearTestList();
    runHistory = [];
    qualityHistory = [];
    els.runList.innerHTML = '';

    connectSession();
    connectCamera();

    setTimeout(() => {
        if (sessionWs && sessionWs.readyState === WebSocket.OPEN) {
            sessionWs.send(JSON.stringify({
                command: 'start',
                sub: subVal,
                ses: sesVal,
                port: els.port.value || null,
                n_sensors: parseInt(els.nSensors.value, 10) || 4,
                dry_run: els.dryRun.checked,
                skip_precollect: els.skipPrecollect.checked,
                n_reps: parseInt(els.nReps.value, 10) || 3,
                record_duration: parseFloat(els.recordDuration.value) || null,
                prep_duration: parseFloat(els.prepDuration.value) || null,
                rest_duration: parseFloat(els.restDuration.value) || null,
                include_freeform: els.includeFreeform.checked,
                camera_index: parseInt(els.cameraIndex ? els.cameraIndex.value : '0', 10) || 0,
            }));
            els.startBtn.disabled = true;
            els.stopBtn.disabled = false;
            log('Sent start command');
        } else {
            log('Session WebSocket not ready');
        }
    }, 300);
}

// ── Subject / session auto-suggest ───────────────────────────────────────────

async function suggestNextSubject() {
    try {
        const resp = await fetch('/api/next_subject');
        const data = await resp.json();
        if (data.subject) {
            els.sub.value = data.subject;
            log(`Suggested next subject: ${data.subject}`);
            suggestNextSession(data.subject);
        }
    } catch (e) {
        log('Could not fetch next subject: ' + e.message);
    }
}

async function suggestNextSession(sub) {
    if (!sub) return;
    try {
        const resp = await fetch(`/api/next_session?sub=${encodeURIComponent(sub)}`);
        const data = await resp.json();
        if (data.session) {
            els.ses.value = data.session;
            log(`Suggested next session for ${sub}: ${data.session}`);
        }
    } catch (e) {
        log('Could not fetch next session: ' + e.message);
    }
}

async function loadExistingSubjects() {
    try {
        const resp = await fetch('/api/subjects');
        const data = await resp.json();
        if (data.subjects) {
            els.existingSubjectSelect.innerHTML = '<option value="">-- Select subject --</option>';
            for (const s of data.subjects) {
                const opt = document.createElement('option');
                opt.value = s;
                opt.textContent = s;
                els.existingSubjectSelect.appendChild(opt);
            }
            if (data.subjects.length === 0) {
                log('No existing subjects found in data/raw/.');
            }
        }
    } catch (e) {
        log('Could not load subjects: ' + e.message);
    }
}

function switchSubjectMode(mode) {
    if (mode === 'new') {
        els.newSubjectRow.style.display = '';
        els.existingSubjectRow.style.display = 'none';
        // Subject ID is always auto-generated; existing mode only selects it.
        els.sub.readOnly = true;
        // Auto-suggest next subject on switch to "new" mode
        suggestNextSubject();
    } else {
        els.newSubjectRow.style.display = 'none';
        els.existingSubjectRow.style.display = '';
        els.sub.readOnly = true;
        loadExistingSubjects();
    }
}

function stopSession() {
    sendCommand('stop');
    els.stopBtn.disabled = true;
}

function restartSession() {
    els.startBtn.disabled = false;
    els.stopBtn.disabled = true;
    els.beginGatePanel.style.display = 'block';
    els.collectActive.style.display = 'none';
    showScreen('setup');
    connectCamera();
}

// ── Dry-run hint toggle ──────────────────────────────────────────────────────

function updateDryRunHint() {
    if (els.dryRun.checked) {
        els.dryRunHint.textContent = 'Dry run is ON \u2014 using simulated sensor and camera data. No hardware needed.';
        els.dryRunHint.style.color = 'var(--warning)';
    } else {
        els.dryRunHint.textContent = 'Dry run is OFF \u2014 enter a serial port and connect a camera for live data.';
        els.dryRunHint.style.color = 'var(--muted)';
    }
}

// ── Init ─────────────────────────────────────────────────────────────────────

function init() {
    createFsrBars(els.fsrBars, 'collect');
    createFsrBars(els.testFsrBars, 'test');
    createFsrBars(els.setupFsrBars, 'setup');

    els.startBtn.addEventListener('click', startSession);
    els.stopBtn.addEventListener('click', stopSession);
    els.restartBtn.addEventListener('click', restartSession);

    els.testReadyBtn.addEventListener('click', () => {
        sendCommand('test_ready');
        els.testReadyBtn.style.display = 'none';
        log('Sent test_ready command');
    });

    els.beginCollectionBtn.addEventListener('click', () => {
        sendCommand('begin_collection');
        log('Sent begin_collection command');
    });

    els.overrideBtn.addEventListener('click', () => {
        sendCommand('begin_collection');
        log('Skipped to collection by operator');
    });

    if (els.retryBtn) {
        els.retryBtn.addEventListener('click', () => {
            sendCommand('retry_tests');
            els.retryBtn.style.display = 'none';
            log('Sent retry_tests command — re-running failed tests');
        });
    }

    function goBackToSetup() {
        if (!confirm('Go back to Setup? This will stop the current session and any unsaved data will be lost.')) {
            return;
        }
        navigatingBack = true;
        sendCommand('stop');
        stopFsrPolling();
        stopCamera();
        els.startBtn.disabled = false;
        els.stopBtn.disabled = true;
        els.beginGatePanel.style.display = 'block';
        els.collectActive.style.display = 'none';
        if (els.retryBtn) els.retryBtn.style.display = 'none';
        showScreen('setup');
        connectCamera();
        // Do NOT reconnect the session WebSocket here. We wait for the
        // session_ended handshake from the backend, then the next Start
        // Session will reconnect if needed.
        log('Waiting for session_ended handshake before returning to Setup');
    }

    if (els.backToSetupBtn) {
        els.backToSetupBtn.addEventListener('click', goBackToSetup);
    }

    if (els.collectBackToSetupBtn) {
        els.collectBackToSetupBtn.addEventListener('click', goBackToSetup);
    }

    els.dryRun.addEventListener('change', updateDryRunHint);

    if (els.scanCamerasBtn) {
        els.scanCamerasBtn.addEventListener('click', scanCameras);
    }
    if (els.scanPortsBtn) {
        els.scanPortsBtn.addEventListener('click', scanPorts);
    }
    if (els.calibrateLedRoiBtn) {
        els.calibrateLedRoiBtn.addEventListener('click', calibrateLedRoi);
    }

    // Camera tracking settings
    if (els.applyTrackingBtn) {
        els.applyTrackingBtn.addEventListener('click', applyCameraSettings);
    }
    // Live-update range value labels
    if (els.camDetectionConf) {
        els.camDetectionConf.addEventListener('input', () => {
            els.camDetectionConfVal.textContent = parseFloat(els.camDetectionConf.value).toFixed(2);
        });
    }
    if (els.camPresenceConf) {
        els.camPresenceConf.addEventListener('input', () => {
            els.camPresenceConfVal.textContent = parseFloat(els.camPresenceConf.value).toFixed(2);
        });
    }
    if (els.camTrackingConf) {
        els.camTrackingConf.addEventListener('input', () => {
            els.camTrackingConfVal.textContent = parseFloat(els.camTrackingConf.value).toFixed(2);
        });
    }
    loadCameraSettings();

    updateDryRunHint();
    showScreen('setup');
    connectCamera();
    connectSession();
    // Auto-scan serial ports on page load so the dropdown is populated
    scanPorts();
    // Load LED ROI status if previously saved
    loadLedRoiStatus();
    // Start FSR polling so bars move on the Setup screen immediately
    startFsrPolling();

    // Subject/session auto-suggest wiring
    els.subjectModeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => switchSubjectMode(e.target.value));
    });
    if (els.suggestSubjectBtn) {
        els.suggestSubjectBtn.addEventListener('click', suggestNextSubject);
    }
    if (els.suggestSessionBtn) {
        els.suggestSessionBtn.addEventListener('click', () => suggestNextSession(els.sub.value.trim()));
    }
    if (els.refreshSubjectsBtn) {
        els.refreshSubjectsBtn.addEventListener('click', loadExistingSubjects);
    }
    if (els.existingSubjectSelect) {
        els.existingSubjectSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                els.sub.value = e.target.value;
                suggestNextSession(e.target.value);
            }
        });
    }
    if (els.sub) {
        els.sub.addEventListener('blur', () => {
            if (els.sub.value.trim()) suggestNextSession(els.sub.value.trim());
        });
    }
    // Auto-suggest next subject on page load (New Subject mode is default)
    suggestNextSubject();
}

init();
