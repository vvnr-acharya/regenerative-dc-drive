(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const isStaticDemo = /\.github\.io$/i.test(window.location.hostname);
  const state = {
    socket: null,
    reconnectTimer: null,
    reconnectDelay: 1000,
    demoTimer: null,
    demoRpm: 0,
    latest: null,
    samples: [],
    chartSamples: [],
    lastSampleAt: 0,
    optimalDuty: null,
    motor: null,
    driveControl: { throttle: 0, brakeDuty: 0 },
    brakingResult: null,
    optimizerResult: null,
    adminOverview: null,
    threeLoadStarted: false
  };

  const pageNames = {
    dashboard: "Overview",
    braking: "Braking distance",
    optimizer: "Regen optimizer",
    analytics: "Live telemetry",
    safety: "Safety & health",
    admin: "Admin portal"
  };

  function navigate() {
    const path = window.location.pathname.replace(/\/+$/, "").split("/").pop();
    const requested = window.location.hash.slice(1) || path || "dashboard";
    if (isStaticDemo && requested === "admin") {
      window.location.hash = "dashboard";
      return;
    }
    if (requested === "admin" && path !== "admin") {
      window.location.replace("/admin");
      return;
    }
    const route = pageNames[requested] ? requested : "dashboard";
    document.querySelectorAll("[data-page]").forEach((page) => {
      const active = page.dataset.page === route;
      page.classList.toggle("active", active);
      page.setAttribute("aria-hidden", String(!active));
    });
    document.querySelectorAll("[data-view]").forEach((link) => {
      const active = link.dataset.view === route;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    $("#breadcrumb").textContent = `Workspace / ${pageNames[route]}`;
    if (state.motor) resizeMotor();
    window.requestAnimationFrame(() => {
      drawCharts();
    });
    if (route === "admin") refreshAdmin();
  }

  document.querySelectorAll("[data-view]").forEach((link) => {
    if (isStaticDemo && link.dataset.view === "admin") {
      link.hidden = true;
      link.style.display = "none";
      return;
    }
    link.addEventListener("click", (event) => {
      if (link.dataset.view === "admin") return;
      event.preventDefault();
      window.history.pushState({}, "", link.getAttribute("href"));
      navigate();
    });
  });

  function setConnection(connected, label) {
    const pill = $("#connection-status");
    pill.classList.toggle("connected", connected);
    $("#connection-label").textContent = label;
  }

  function send(payload) {
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
      state.socket.send(JSON.stringify(payload));
      return true;
    }
    return false;
  }

  function connectSocket() {
    if (isStaticDemo) {
      setConnection(false, "Static demo");
      $("#mode-label").textContent = "Browser simulation";
      $("#operating-mode").disabled = true;
      $("#page-optimizer .subtitle").textContent = "Explore the interface. ML predictions are available in the live app.";
      $("#optimizer-form .card-kicker").textContent = "Live backend required";
      $("#optimizer-form .note").textContent = "This GitHub Pages preview runs in your browser only. The live ML optimizer requires the FastAPI backend.";
      state.demoTimer = window.setInterval(updateDemoTelemetry, 1000);
      updateDemoTelemetry();
      return;
    }
    if (state.socket && (state.socket.readyState === WebSocket.OPEN || state.socket.readyState === WebSocket.CONNECTING)) return;
    const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
    state.socket = new WebSocket(`${scheme}//${window.location.host}/ws/telemetry`);
    setConnection(false, "Connecting");

    state.socket.addEventListener("open", () => {
      state.reconnectDelay = 1000;
      setConnection(true, "Live");
      send({ type: "mode", mode: $("#operating-mode").value });
      send({
        type: "control",
        throttle: state.driveControl.throttle,
        brake_duty: state.driveControl.brakeDuty
      });
    });
    state.socket.addEventListener("message", (event) => {
      try {
        const telemetry = JSON.parse(event.data);
        if (telemetry && typeof telemetry === "object") updateTelemetry(telemetry);
      } catch {
        setConnection(false, "Bad telemetry");
      }
    });
    state.socket.addEventListener("close", () => {
      setConnection(false, "Reconnecting");
      clearTimeout(state.reconnectTimer);
      state.reconnectTimer = window.setTimeout(connectSocket, state.reconnectDelay);
      state.reconnectDelay = Math.min(state.reconnectDelay * 1.7, 10000);
    });
    state.socket.addEventListener("error", () => setConnection(false, "Unavailable"));
  }

  function updateDemoTelemetry() {
    const { throttle, brakeDuty } = state.driveControl;
    const targetRpm = throttle * 3200;
    if (brakeDuty > 0) state.demoRpm = Math.max(0, state.demoRpm - brakeDuty * 900);
    else state.demoRpm += (targetRpm - state.demoRpm) * (throttle > 0 ? 0.28 : 0.015);
    updateTelemetry({
      timestamp: Date.now() / 1000,
      mode: "simulation",
      rpm: state.demoRpm,
      v_bat: 12 + brakeDuty * 1.8,
      i_shunt: throttle * 4 + brakeDuty * 3,
      temp: 35 + throttle * 5 + brakeDuty * 3,
      duty: brakeDuty,
      throttle,
      peak_current_a: throttle * 4 + brakeDuty * 3,
      peak_kickback_voltage_v: 12 + brakeDuty * 1.8
    });
  }

  function updateTelemetry(data) {
    state.latest = data;
    const rpm = Number(data.rpm) || 0;
    const voltage = Number(data.v_bat) || 0;
    const current = Number(data.i_shunt) || 0;
    const temperature = Number(data.temp) || 0;
    const duty = Number(data.duty) || 0;
    const throttle = Number(data.throttle) || 0;
    const mode = data.mode === "hardware" ? "ESP32 hardware" : "Simulation";
    $("#mode-label").textContent = mode;
    $("#operating-mode").value = data.mode === "hardware" ? "hardware" : "simulation";

    const regenActive = duty > 0.05;
    const motorActive = !regenActive && throttle > 0.01;
    const driveState = regenActive ? "Regenerative braking" : motorActive ? "Motoring" : "Coasting";
    $("#drive-state").textContent = driveState;
    $("#motor-state").textContent = driveState;
    setDriveMode(regenActive ? "regen" : motorActive ? "motoring" : "coasting");

    const now = Date.now();
    if (now - state.lastSampleAt >= 1000) {
      $("#metric-rpm").textContent = rpm.toFixed(0);
      $("#metric-voltage").textContent = voltage.toFixed(2);
      $("#metric-current").textContent = current.toFixed(2);
      $("#metric-temperature").textContent = temperature.toFixed(1);
      $("#safety-current").textContent = Number(data.peak_current_a ?? current).toFixed(2);
      $("#safety-kickback").textContent = Number(data.peak_kickback_voltage_v ?? voltage).toFixed(2);
      $("#safety-temperature").textContent = temperature.toFixed(1);

      const sample = {
        timestamp: Number(data.timestamp) || now / 1000,
        mode: data.mode === "hardware" ? "Hardware" : "Simulation",
        rpm,
        voltage,
        current,
        temperature,
        duty
      };
      state.samples.unshift(sample);
      state.samples = state.samples.slice(0, 20);
      state.chartSamples.push(sample);
      state.chartSamples = state.chartSamples.slice(-60);
      state.lastSampleAt = now;
      renderTelemetryTable();
      drawCharts();
      renderAdmin();
      const prediction = data.ml_predictions;
      if (prediction) {
        const energy = Number(prediction.energy_recovered_joules);
        const label = String(prediction.safety_label || "");
        if (Number.isFinite(energy)) {
          $("#home-energy").textContent = `${energy.toFixed(3)} J predicted recovery`;
          $("#home-insight").textContent = label || "Model estimate updated from the latest drive telemetry.";
        }
        updateSafetyBanner(prediction, data);
      }
    }
  }

  function renderTelemetryTable() {
    const table = $("#telemetry-table");
    table.replaceChildren();
    if (!state.samples.length) {
      const row = table.insertRow();
      const cell = row.insertCell();
      cell.colSpan = 7;
      cell.className = "empty-state";
      cell.textContent = "Waiting for telemetry…";
      return;
    }
    state.samples.forEach((sample) => {
      const row = table.insertRow();
      const values = [
        new Date(sample.timestamp * 1000).toLocaleTimeString(),
        sample.mode,
        sample.rpm.toFixed(0),
        `${sample.voltage.toFixed(2)} V`,
        `${sample.current.toFixed(2)} A`,
        `${sample.temperature.toFixed(1)} °C`,
        `${(sample.duty * 100).toFixed(0)}%`
      ];
      values.forEach((value) => { row.insertCell().textContent = value; });
    });
  }

  function drawCharts() {
    drawChart($("#dashboard-chart"));
    drawChart($("#analytics-chart"));
    drawChart($("#admin-chart"));
  }

  function drawChart(canvas) {
    if (!canvas || !canvas.isConnected) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const width = rect.width;
    const height = rect.height;
    const left = 31;
    const right = 8;
    const top = 7;
    const bottom = 20;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    context.clearRect(0, 0, width, height);
    context.font = "10px system-ui, sans-serif";
    context.fillStyle = "#9aa39b";
    context.strokeStyle = "#eceee6";
    context.lineWidth = 1;
    for (let index = 0; index < 4; index += 1) {
      const y = top + (plotHeight * index) / 3;
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(width - right, y);
      context.stroke();
    }
    context.fillText("recent", left, height - 4);
    context.fillText("now", width - right - 22, height - 4);

    if (state.chartSamples.length < 2) {
      context.fillStyle = "#8b968e";
      context.textAlign = "center";
      context.fillText("Waiting for live samples…", width / 2, height / 2);
      context.textAlign = "left";
      return;
    }

    const drawSeries = (key, color) => {
      const values = state.chartSamples.map((sample) => sample[key]);
      let minimum = Math.min(...values);
      let maximum = Math.max(...values);
      if (maximum - minimum < 0.1) {
        const padding = key === "voltage" ? 1 : 0.5;
        minimum -= padding;
        maximum += padding;
      }
      context.beginPath();
      values.forEach((value, index) => {
        const x = left + (plotWidth * index) / (values.length - 1);
        const y = top + plotHeight - ((value - minimum) / (maximum - minimum)) * plotHeight;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.lineJoin = "round";
      context.stroke();
    };
    drawSeries("voltage", "#69a987");
    drawSeries("current", "#dc9a77");
  }

  function updateSafetyBanner(prediction, telemetry) {
    const banner = $("#safety-banner");
    const safetyClass = Number(prediction.safety_class);
    const temperature = Number(telemetry.temp) || 0;
    const current = Number(telemetry.peak_current_a ?? telemetry.i_shunt) || 0;
    const kickback = Number(telemetry.peak_kickback_voltage_v ?? telemetry.v_bat) || 0;
    const isHazard = safetyClass === 2;
    const isWatch = isHazard || temperature >= 80 || current >= 15 || kickback >= 18;
    banner.classList.toggle("hazard", isHazard);
    banner.classList.toggle("warning", isWatch && !isHazard);
    $("#safety-class").textContent = prediction.safety_label || `Class ${safetyClass}`;
    $("#safety-title").textContent = isHazard ? "Model flagged a hazard condition" : isWatch ? "Review operating conditions" : "Signals are within the model's normal range";
    $("#safety-description").textContent = prediction.safety_label
      ? `${prediction.safety_label}. This is a model advisory based on the latest telemetry.`
      : "Waiting for a model classification.";
  }

  function setDriveMode(mode) {
    if (!state.motor) return;
    if (state.motor.mode === mode) return;
    state.motor.mode = mode;
    const flow = $("#motor-flow");
    flow.classList.toggle("regen", mode === "regen");
    const descriptions = {
      motoring: ["Source → MOSFET bridge → motor", "Motoring"],
      regen: ["Motor → MOSFET bridge → capacitor", "Regeneration"],
      coasting: ["No active current path", "Idle"]
    };
    const [path, title] = descriptions[mode] || descriptions.coasting;
    flow.innerHTML = `Current path · <strong>${path}</strong>`;
    state.motor.pathTitle = title;
    const activeSwitches = mode === "motoring" ? [0, 3] : mode === "regen" ? [1, 2] : [];
    state.motor.switches.forEach((entry, index) => {
      const active = activeSwitches.includes(index);
      entry.body.material.emissive.setHex(active ? (mode === "regen" ? 0x9a5836 : 0x3c7b58) : 0x000000);
      entry.body.material.emissiveIntensity = active ? .38 : 0;
      entry.led.material.color.setHex(active ? (mode === "regen" ? 0xe59a71 : 0x72bb8c) : 0xc5c8bb);
    });
  }

  function setupControls() {
    const throttle = $("#throttle");
    const brake = $("#brake");
    const updateControls = () => {
      const throttleValue = Number(throttle.value);
      const brakeValue = Number(brake.value);
      $("#throttle-value").textContent = `${throttleValue}%`;
      $("#brake-value").textContent = `${brakeValue}%`;
      $("#drive-state").textContent = brakeValue > 5 ? "Regenerative braking" : throttleValue > 0 ? "Motoring" : "Coasting";
      $("#motor-state").textContent = $("#drive-state").textContent;
      state.driveControl = { throttle: throttleValue / 100, brakeDuty: brakeValue / 100 };
      setDriveMode(brakeValue > 5 ? "regen" : throttleValue > 0 ? "motoring" : "coasting");
      send({ type: "control", throttle: throttleValue / 100, brake_duty: brakeValue / 100 });
    };
    throttle.addEventListener("input", () => {
      if (Number(throttle.value) > 0) brake.value = "0";
      updateControls();
    });
    brake.addEventListener("input", () => {
      if (Number(brake.value) > 0) throttle.value = "0";
      updateControls();
    });
    $("#operating-mode").addEventListener("change", (event) => send({ type: "mode", mode: event.target.value }));
    $("#coast-button").addEventListener("click", () => {
      throttle.value = "0";
      brake.value = "0";
      updateControls();
    });
    $("#stop-button").addEventListener("click", () => {
      throttle.value = "0";
      brake.value = "90";
      updateControls();
    });
    $("#reset-simulation").addEventListener("click", () => {
      throttle.value = "0";
      brake.value = "0";
      updateControls();
      send({ type: "reset" });
    });
    updateControls();
  }

  function setupBrakingCalculator() {
    const calculate = () => {
      const error = $("#braking-error");
      error.textContent = "";
      const start = Number($("#speed-start").value);
      const end = Number($("#speed-end").value);
      const deceleration = Number($("#deceleration").value);
      const mass = Number($("#vehicle-mass").value);
      const efficiency = Number($("#regen-efficiency").value);
      if (![start, end, deceleration, mass, efficiency].every(Number.isFinite) || start < end || deceleration <= 0 || mass <= 0 || efficiency < 0 || efficiency > 100) {
        error.textContent = "Check that speeds and values are valid, and target speed is no higher than start.";
        return;
      }
      const startMps = start / 3.6;
      const endMps = end / 3.6;
      const distance = Math.max(0, (startMps ** 2 - endMps ** 2) / (2 * deceleration));
      const time = Math.max(0, (startMps - endMps) / deceleration);
      const energy = Math.max(0, 0.5 * mass * (startMps ** 2 - endMps ** 2));
      state.brakingResult = {
        start,
        end,
        deceleration,
        mass,
        efficiency,
        distance,
        time,
        energy,
        recoveredEnergy: energy * efficiency / 100
      };
      $("#brake-distance").textContent = distance.toFixed(1);
      $("#brake-time").textContent = `${time.toFixed(2)} s`;
      $("#brake-energy").textContent = `${(energy / 1000).toFixed(2)} kJ`;
      $("#brake-recovered").textContent = `${(energy * efficiency / 100 / 1000).toFixed(2)} kJ`;
      renderAdmin();
    };
    $("#braking-form").addEventListener("submit", (event) => {
      event.preventDefault();
      calculate();
    });
    calculate();
  }

  async function runOptimizer(event) {
    event.preventDefault();
    const error = $("#optimizer-error");
    if (isStaticDemo) {
      error.textContent = "ML predictions need the backend. Open the live tunnel for optimizer results.";
      return;
    }
    const button = $("#optimize-button");
    error.textContent = "";
    button.disabled = true;
    button.textContent = "Calculating…";
    try {
      const body = {
        initial_rpm: Number($("#opt-rpm").value),
        pwm_brake_duty: Number($("#opt-duty").value) / 100,
        bus_voltage_init: Number($("#opt-voltage").value),
        motor_temp_c: Number($("#opt-temp").value)
      };
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "The optimizer request failed.");
      const prediction = result.current_prediction || {};
      const optimization = result.optimization || {};
      const energy = Number(optimization.expected_energy_recovered_j);
      const duty = Number(optimization.optimal_brake_duty);
      const currentEnergy = Number(prediction.energy_recovered_joules);
      const stoppingTime = Number(prediction.stopping_time_ms);
      if (![energy, duty, currentEnergy, stoppingTime].every(Number.isFinite)) throw new Error("The optimizer returned incomplete results.");
      state.optimizerResult = { request: body, result };
      state.optimalDuty = duty;
      $("#optimal-duty").textContent = (duty * 100).toFixed(0);
      $("#optimal-energy").textContent = energy.toFixed(3);
      $("#current-energy").textContent = `${currentEnergy.toFixed(3)} J`;
      $("#optimizer-stop-time").textContent = `${stoppingTime.toFixed(1)} ms`;
      const batteryWh = Number($("#battery-capacity").value);
      $("#battery-share").textContent = batteryWh > 0 ? `${(energy / (batteryWh * 3600) * 100).toFixed(4)}%` : "—";
      const safety = $("#optimizer-safety");
      safety.textContent = prediction.safety_label || "Model estimate";
      safety.className = "safety-pill";
      if (Number(prediction.safety_class) === 1) safety.classList.add("safe");
      else if (Number(prediction.safety_class) === 2) safety.classList.add("hazard");
      else safety.classList.add("warning");
      renderDutySweep(optimization.duty_sweep || [], duty);
      $("#apply-optimal").disabled = false;
      renderAdmin();
    } catch (requestError) {
      error.textContent = requestError instanceof Error ? requestError.message : "Unable to calculate an estimate.";
    } finally {
      button.disabled = false;
      button.textContent = "Optimize recovery";
    }
  }

  function renderDutySweep(items, optimalDuty) {
    const container = $("#duty-sweep");
    container.replaceChildren();
    const values = items.map((item) => Number(item.energy_recovered_j)).filter(Number.isFinite);
    const max = Math.max(0, ...values);
    if (!items.length) {
      container.innerHTML = '<p class="empty-state">No sweep results were returned.</p>';
      return;
    }
    items.forEach((item) => {
      const duty = Number(item.duty);
      const energy = Number(item.energy_recovered_j);
      if (!Number.isFinite(duty) || !Number.isFinite(energy)) return;
      const row = document.createElement("div");
      row.className = `sweep-row${Math.abs(duty - optimalDuty) < 0.001 ? " optimal" : ""}`;
      const label = document.createElement("span");
      label.textContent = `${Math.round(duty * 100)}%`;
      const track = document.createElement("div");
      track.className = "sweep-track";
      const bar = document.createElement("div");
      bar.className = "sweep-bar";
      bar.style.width = `${Math.max(3, max > 0 ? (Math.max(0, energy) / max) * 100 : 3)}%`;
      track.append(bar);
      const amount = document.createElement("span");
      amount.className = "sweep-energy";
      amount.textContent = `${energy.toFixed(3)} J`;
      row.append(label, track, amount);
      container.append(row);
    });
  }

  function setupOptimizer() {
    $("#optimizer-form").addEventListener("submit", runOptimizer);
    $("#apply-optimal").addEventListener("click", () => {
      if (!Number.isFinite(state.optimalDuty)) return;
      const brake = Math.min(90, Math.max(0, Math.round(state.optimalDuty * 100)));
      $("#brake").value = String(brake);
      $("#throttle").value = "0";
      $("#brake").dispatchEvent(new Event("input", { bubbles: true }));
      window.location.hash = "dashboard";
    });
  }

  async function refreshHealth() {
    if (isStaticDemo) {
      $("#backend-health").textContent = "Static demo";
      $("#model-health").textContent = "Backend required";
      return;
    }
    $("#backend-health").textContent = "Checking…";
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      const status = await response.json();
      $("#backend-health").textContent = status.status === "online" ? "Online" : "Unavailable";
      $("#model-health").textContent = status.models_loaded ? "Loaded" : "Not loaded";
      if (status.latest_telemetry && !state.latest) updateTelemetry(status.latest_telemetry);
    } catch {
      $("#backend-health").textContent = "Unavailable";
      $("#model-health").textContent = "Unknown";
    }
  }

  async function refreshAdmin() {
    const backend = $("#admin-backend");
    if (!backend) return;
    try {
      const response = await fetch("/api/admin/overview", { cache: "no-store" });
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      state.adminOverview = await response.json();
      if (state.adminOverview.latest_telemetry) {
        updateTelemetry(state.adminOverview.latest_telemetry);
      }
      renderAdmin();
    } catch (error) {
      backend.textContent = "Unavailable";
      const uptime = $("#admin-uptime");
      if (uptime) uptime.textContent = error instanceof Error ? error.message : "Diagnostics request failed";
    }
  }

  function renderAdminAnalytics(analytics) {
    const container = $("#admin-analytics-methods");
    const context = $("#admin-training-context");
    if (!container || !analytics) return;
    container.replaceChildren();

    const addText = (parent, label, value) => {
      const paragraph = document.createElement("p");
      const strong = document.createElement("strong");
      strong.textContent = `${label}: `;
      paragraph.append(strong, document.createTextNode(String(value)));
      parent.append(paragraph);
    };
    const makeCard = (title) => {
      const card = document.createElement("article");
      card.className = "algorithm-card";
      const heading = document.createElement("h3");
      heading.textContent = title;
      card.append(heading);
      container.append(card);
      return card;
    };

    const preprocessing = analytics.preprocessing || {};
    const preprocessingCard = makeCard(`${preprocessing.algorithm || "Feature scaler"} · preprocessing`);
    addText(preprocessingCard, "Where", preprocessing.where_used || "Unavailable");
    addText(preprocessingCard, "Why", preprocessing.why || "Unavailable");

    (analytics.estimators || []).forEach((estimator) => {
      const card = makeCard(estimator.algorithm || "Unavailable estimator");
      addText(card, "Predicts", estimator.target || "Unavailable");
      addText(card, "Used in", estimator.where_used || "Unavailable");
      addText(card, "Why selected", estimator.why || "Unavailable");
      const parameters = Object.entries(estimator.parameters || {})
        .map(([name, value]) => `${name}=${value}`)
        .join(", ");
      if (parameters) addText(card, "Loaded parameters", parameters);
      if (estimator.feature_importances?.length) {
        const label = document.createElement("p");
        label.textContent = "Top fitted feature importances (impurity-based, not causal):";
        const list = document.createElement("ul");
        estimator.feature_importances.slice(0, 4).forEach((item) => {
          const entry = document.createElement("li");
          entry.textContent = `${item.feature}: ${(Number(item.importance) * 100).toFixed(1)}%`;
          list.append(entry);
        });
        card.append(label, list);
      }
    });

    const optimizer = analytics.optimizer || {};
    const optimizerCard = makeCard(optimizer.algorithm || "Optimizer");
    addText(optimizerCard, "Where", optimizer.where_used || "Unavailable");
    addText(optimizerCard, "Search", optimizer.search_space || "Unavailable");
    addText(optimizerCard, "Objective / constraint", optimizer.objective || "Unavailable");
    addText(optimizerCard, "Why this method", optimizer.why || "Unavailable");

    const training = analytics.training_data || {};
    const trainingCard = makeCard("Training data & evaluation");
    addText(trainingCard, "Dataset", training.source || "Unavailable");
    addText(trainingCard, "Provenance", training.provenance || "Unavailable");
    addText(trainingCard, "Split", training.split || "Unavailable");
    addText(trainingCard, "Evaluation", training.evaluation || "Unavailable");
    addText(trainingCard, "Metric availability", training.metrics_note || "Unavailable");
    addText(trainingCard, "Model inputs", (training.features || []).join(", ") || "Unavailable");

    const notUsedCard = makeCard("Methods not in this pipeline");
    const notUsed = document.createElement("p");
    notUsed.textContent = analytics.not_used || "Unavailable";
    notUsedCard.append(notUsed);

    if (context) {
      context.replaceChildren();
      const note = document.createElement("p");
      note.className = "admin-explain";
      note.textContent = "Feature-importance values describe how fitted tree splits used the inputs; they do not prove causal effects. All model predictions are trained on simulation-generated data and are advisory.";
      context.append(note);
    }
  }

  function renderAdmin() {
    if (!$("#admin-backend")) return;
    const overview = state.adminOverview;
    const telemetry = state.latest || overview?.latest_telemetry || null;
    if (overview) {
      $("#admin-backend").textContent = overview.status === "online" ? "Online" : "Unavailable";
      const uptimeSeconds = Number(overview.uptime_seconds);
      const uptime = Number.isFinite(uptimeSeconds) ? uptimeSeconds : 0;
      $("#admin-uptime").textContent = `Uptime ${Math.floor(uptime / 3600)}h ${Math.floor((uptime % 3600) / 60)}m ${Math.floor(uptime % 60)}s`;
      $("#admin-mode").textContent = overview.mode === "hardware" ? "ESP32 hardware" : "Simulation";
      $("#admin-clients").textContent = String(overview.active_clients ?? "—");
      $("#admin-models").textContent = overview.models_loaded ? "Loaded" : "Unavailable";
      renderAdminAnalytics(overview.analytics);
      $("#admin-throttle").textContent = `${((Number(overview.controls?.throttle) || 0) * 100).toFixed(0)}%`;
      $("#admin-brake-duty").textContent = `${((Number(overview.controls?.brake_duty) || 0) * 100).toFixed(0)}%`;

      const rates = overview.rates || {};
      $("#admin-rate-grid").replaceChildren();
      [
        ["Physics loop", `${rates.simulation_hz ?? "—"} Hz`],
        ["ESP32 telemetry source", `${rates.hardware_telemetry_hz ?? "—"} Hz`],
        ["Dashboard readout", `${((Number(rates.dashboard_readout_interval_ms) || 0) / 1000).toFixed(1)} s`]
      ].forEach(([label, value]) => {
        const tile = document.createElement("div");
        tile.className = "status-tile";
        const caption = document.createElement("span");
        caption.textContent = label;
        const result = document.createElement("strong");
        result.textContent = value;
        tile.append(caption, result);
        $("#admin-rate-grid").append(tile);
      });

      const artifacts = $("#admin-artifacts");
      artifacts.replaceChildren();
      (overview.model_artifacts || []).forEach((artifact) => {
        const row = artifacts.insertRow();
        [
          artifact.name,
          artifact.role,
          artifact.loaded ? "Loaded" : artifact.present ? "Present · not loaded" : "Missing",
          Number.isFinite(Number(artifact.size_bytes)) ? `${(Number(artifact.size_bytes) / 1024).toFixed(1)} KB` : "—"
        ].forEach((value) => { row.insertCell().textContent = String(value); });
      });

      const parameters = $("#admin-physics-parameters");
      parameters.replaceChildren();
      Object.entries(overview.simulation_parameters || {}).forEach(([key, value]) => {
        const row = document.createElement("div");
        row.className = "result-row";
        const label = document.createElement("span");
        label.textContent = key.replaceAll("_", " ");
        const result = document.createElement("strong");
        result.textContent = String(value);
        row.append(label, result);
        parameters.append(row);
      });
    }

    if (telemetry) {
      const rpm = Number(telemetry.rpm) || 0;
      const voltage = Number(telemetry.v_bat) || 0;
      const current = Number(telemetry.i_shunt) || 0;
      const temperature = Number(telemetry.temp) || 0;
      const duty = Number(telemetry.duty) || 0;
      const throttle = Number(telemetry.throttle) || 0;
      const regenerating = duty > .05;
      $("#admin-drive-state").textContent = regenerating
        ? "Regeneration · motor → bridge → capacitor"
        : throttle > .01 ? "Motoring · source → bridge → motor" : "Coasting · no commanded path";
      $("#admin-speed-bus").textContent = `${rpm.toFixed(0)} rpm · ${voltage.toFixed(2)} V`;
      $("#admin-safety-electrical").textContent =
        `${Number(telemetry.peak_current_a ?? current).toFixed(2)} A · ${Number(telemetry.peak_kickback_voltage_v ?? voltage).toFixed(2)} V`;
      $("#admin-safety-thermal").textContent =
        `${temperature.toFixed(1)} °C · ${Number(telemetry.mean_current_ripple ?? 0).toFixed(4)} A`;
      $("#admin-sample-count").textContent = String(state.samples.length);
      $("#admin-packet-mode").textContent = telemetry.mode === "hardware" ? "ESP32 hardware" : "Simulation";
      const packetTime = Number(telemetry.timestamp);
      $("#admin-packet-age").textContent = Number.isFinite(packetTime)
        ? `${Math.max(0, (Date.now() / 1000 - packetTime)).toFixed(1)} s`
        : "Unavailable";
      const ml = telemetry.ml_predictions;
      $("#admin-safety-class").textContent = ml?.safety_label || "Waiting for model output";
      $("#admin-raw-telemetry").textContent = JSON.stringify(telemetry, null, 2);
    } else {
      $("#admin-drive-state").textContent = "Waiting for telemetry";
      $("#admin-speed-bus").textContent = "—";
      $("#admin-safety-class").textContent = "Waiting for telemetry";
      $("#admin-raw-telemetry").textContent = "Waiting for telemetry…";
    }

    const telemetryRows = $("#admin-telemetry-table");
    telemetryRows.replaceChildren();
    if (state.samples.length) {
      state.samples.forEach((sample) => {
        const row = telemetryRows.insertRow();
        [
          new Date(sample.timestamp * 1000).toLocaleTimeString(),
          sample.mode,
          sample.rpm.toFixed(0),
          sample.voltage.toFixed(2),
          sample.current.toFixed(2),
          sample.temperature.toFixed(1),
          `${(sample.duty * 100).toFixed(0)}%`
        ].forEach((value) => { row.insertCell().textContent = String(value); });
      });
    } else {
      const row = telemetryRows.insertRow();
      const cell = row.insertCell();
      cell.colSpan = 7;
      cell.className = "empty-state";
      cell.textContent = "Waiting for telemetry…";
    }

    const braking = state.brakingResult;
    $("#admin-braking-speeds").textContent = braking
      ? `${braking.start} → ${braking.end} km/h`
      : "Enter and calculate braking inputs";
    $("#admin-braking-conditions").textContent = braking
      ? `${braking.deceleration} m/s² · ${braking.mass.toLocaleString()} kg · ${braking.efficiency}% capture`
      : "—";
    $("#admin-braking-distance").textContent = braking
      ? `${braking.distance.toFixed(1)} m · ${braking.time.toFixed(2)} s`
      : "—";
    $("#admin-braking-energy").textContent = braking
      ? `${(braking.energy / 1000).toFixed(2)} kJ · ${(braking.recoveredEnergy / 1000).toFixed(2)} kJ recoverable`
      : "—";

    const optimizer = state.optimizerResult;
    if (optimizer) {
      const request = optimizer.request;
      const result = optimizer.result;
      const prediction = result.current_prediction || {};
      const features = result.input_features || {};
      const choice = result.optimization || {};
      $("#admin-optimizer-inputs").textContent =
        `${request.initial_rpm} rpm · ${(request.pwm_brake_duty * 100).toFixed(0)}% · ${request.bus_voltage_init} V · ${request.motor_temp_c} °C`;
      $("#admin-model-features").textContent =
        `${features.peak_current_a ?? "—"} A peak · ${features.peak_kickback_voltage_v ?? "—"} V kickback · ${features.mean_current_ripple ?? "—"} A ripple`;
      $("#admin-current-prediction").textContent =
        `${prediction.energy_recovered_joules ?? "—"} J · ${prediction.stopping_time_ms ?? "—"} ms · ${prediction.safety_label ?? "no class"}`;
      $("#admin-optimal-result").textContent =
        `${(Number(choice.optimal_brake_duty) * 100).toFixed(0)}% · ${Number(choice.expected_energy_recovered_j).toFixed(3)} J · ${(choice.duty_sweep || []).length} sweep points`;
    } else {
      $("#admin-optimizer-inputs").textContent = "No optimizer run yet";
      $("#admin-model-features").textContent = "—";
      $("#admin-current-prediction").textContent = "—";
      $("#admin-optimal-result").textContent = "—";
    }
  }

  function setupSafety() {
    $("#refresh-health").addEventListener("click", refreshHealth);
    $("#refresh-admin").addEventListener("click", refreshAdmin);
    $("#clear-history").addEventListener("click", () => {
      state.samples = [];
      state.chartSamples = [];
      renderTelemetryTable();
      renderAdmin();
      drawCharts();
    });
  }

  function resizeMotor() {
    if (!state.motor) return;
    const { renderer, camera, container } = state.motor;
    if (typeof state.motor.resize === "function") {
      state.motor.resize();
      return;
    }
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (!width || !height) return;
    if (!renderer || !camera) return;
    renderer.setSize(width, height, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function setupMotorFallback(container) {
    state.motor?.resizeObserver?.disconnect();
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", "Animated powertrain and current-flow diagram");
    const context = canvas.getContext("2d");
    if (!context) {
      container.innerHTML = '<div class="empty-state">The powertrain visualization could not initialize. Try a browser with canvas support.</div>';
      return;
    }

    container.replaceChildren(canvas);
    state.motor = { container, canvas, context, renderer: null, camera: null, switches: [], mode: null };
    setDriveMode("coasting");

    let elapsed = 0;
    let previousFrame = 0;
    const paths = {
      motoring: [[.12, .46], [.33, .36], [.41, .43], [.49, .5], [.41, .61], [.32, .68], [.12, .58]],
      regen: [[.53, .46], [.64, .37], [.85, .44], [.85, .59], [.65, .66], [.54, .55]]
    };
    const draw = (timestamp) => {
      elapsed += Math.min((timestamp - (previousFrame || timestamp)) / 1000, .05);
      previousFrame = timestamp;

      const bounds = container.getBoundingClientRect();
      const width = bounds.width;
      const height = bounds.height;
      if (width < 2 || height < 2) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const pixelWidth = Math.round(width * ratio);
      const pixelHeight = Math.round(height * ratio);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const roundedBox = (x, y, w, h, fill, radius = 9) => {
        context.beginPath();
        context.roundRect(x, y, w, h, radius);
        context.fillStyle = fill;
        context.fill();
      };
      const text = (value, x, y, color = "#52645a", size = 10) => {
        context.fillStyle = color;
        context.font = `600 ${size}px system-ui, sans-serif`;
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(value, x, y);
      };

      const mid = height * .52;
      const scale = Math.min(width, height);
      roundedBox(width * .06, height * .25, width * .88, height * .51, "rgba(207,199,180,.55)", 15);
      roundedBox(width * .07, mid - scale * .105, scale * .18, scale * .21, "#83ad91", 12);
      text("DC SOURCE", width * .16, mid - 5, "#456b52", 10);
      text("+     −", width * .16, mid + 12, "#fffefa", 10);

      const fetWidth = Math.max(34, width * .1);
      const fetHeight = Math.max(24, height * .085);
      [[.35, .29], [.59, .29], [.35, .66], [.59, .66]].forEach(([x, y], index) => {
        roundedBox(width * x, height * y, fetWidth, fetHeight, state.motor.mode === "motoring" && [0, 3].includes(index)
          ? "#72aa85"
          : state.motor.mode === "regen" && [1, 2].includes(index) ? "#d69270" : "#9db6a2", 7);
        text(`Q${index + 1}`, width * x + fetWidth / 2, height * y + fetHeight / 2, "#fffefa", 9);
      });

      const motorX = width * .51;
      const motorY = mid;
      const motorRadius = scale * .15;
      context.beginPath();
      context.arc(motorX, motorY, motorRadius, 0, Math.PI * 2);
      context.fillStyle = "#d8c6a2";
      context.fill();
      context.lineWidth = 3;
      context.strokeStyle = "#9db6a2";
      context.stroke();
      context.save();
      context.translate(motorX, motorY);
      context.rotate(elapsed * (Math.max(0, Number(state.latest?.rpm) || 0) * Math.PI / 30 || 1.5));
      for (let blade = 0; blade < 4; blade += 1) {
        context.rotate(Math.PI / 2);
        context.beginPath();
        context.moveTo(0, 0);
        context.lineTo(motorRadius * .72, 0);
        context.strokeStyle = "#c17f5f";
        context.lineWidth = 4;
        context.stroke();
      }
      context.restore();
      text("DC MOTOR", motorX, motorY + motorRadius + 14, "#52645a", 9);

      roundedBox(width * .81, mid - scale * .105, scale * .14, scale * .21, "#c99c6d", 12);
      text("CAP", width * .88, mid - 4, "#fffefa", 10);
      text("CAPACITOR", width * .88, mid + scale * .12, "#765a43", 9);
      text("H-BRIDGE", width * .5, height * .16, "#63746a", 10);

      const activePath = paths[state.motor.mode];
      if (activePath) {
        const points = activePath.map(([x, y]) => [x * width, y * height]);
        context.beginPath();
        points.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
        context.strokeStyle = state.motor.mode === "regen" ? "rgba(201,129,91,.55)" : "rgba(93,157,116,.55)";
        context.lineWidth = 2;
        context.setLineDash([5, 5]);
        context.stroke();
        context.setLineDash([]);

        const progress = (elapsed * .45) % 1;
        for (let dot = 0; dot < 4; dot += 1) {
          const dotProgress = (progress + dot * .23) % 1;
          const dotSegment = dotProgress * (points.length - 1);
          const dotIndex = Math.floor(dotSegment);
          const dotAmount = dotSegment - dotIndex;
          const [ax, ay] = points[dotIndex];
          const [bx, by] = points[Math.min(dotIndex + 1, points.length - 1)];
          context.beginPath();
          context.arc(ax + (bx - ax) * dotAmount, ay + (by - ay) * dotAmount, 3.5, 0, Math.PI * 2);
          context.fillStyle = state.motor.mode === "regen" ? "#c9815b" : "#5d9d74";
          context.fill();
        }
      }
    };
    state.motor.resize = () => draw(performance.now());
    const animate = (timestamp) => {
      draw(timestamp);
      window.requestAnimationFrame(animate);
    };
    draw(performance.now());
    window.requestAnimationFrame(animate);
    if ("ResizeObserver" in window) {
      state.motor.resizeObserver = new ResizeObserver(() => draw(performance.now()));
      state.motor.resizeObserver.observe(container);
    }
  }

  function setupMotor() {
    const container = $("#three-container");
    if (!window.THREE) {
      setupMotorFallback(container);
      if (!state.threeLoadStarted) {
        state.threeLoadStarted = true;
        const library = document.createElement("script");
        library.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
        library.async = true;
        library.addEventListener("load", () => {
          if (window.THREE) setupMotor();
        }, { once: true });
        library.addEventListener("error", () => {
          console.error("Three.js could not be loaded; keeping the animated 2D powertrain twin.");
        }, { once: true });
        document.head.append(library);
      }
      return;
    }
    state.motor?.resizeObserver?.disconnect();
    const THREE = window.THREE;
    let scene;
    let camera;
    let renderer;
    try {
      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
      camera.position.set(0, 5.4, 12);
      camera.lookAt(0, 0, 0);
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (error) {
      console.error("Unable to initialize the WebGL powertrain twin.", error);
      setupMotorFallback(container);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.replaceChildren(renderer.domElement);
    scene.add(new THREE.HemisphereLight(0xfff9eb, 0x9fafa1, .6));
    const keyLight = new THREE.DirectionalLight(0xffffff, .8);
    keyLight.position.set(3, 5, 4);
    scene.add(keyLight);
    const fillLight = new THREE.PointLight(0xe4b99e, .3);
    fillLight.position.set(-4, 1, -3);
    scene.add(fillLight);

    const assembly = new THREE.Group();
    scene.add(assembly);
    const boardMaterial = new THREE.MeshStandardMaterial({ color: 0xcfc7b4, roughness: .82 });
    const sourceMaterial = new THREE.MeshStandardMaterial({ color: 0x83ad91, roughness: .46, metalness: .16 });
    const capacitorMaterial = new THREE.MeshStandardMaterial({ color: 0xc99c6d, roughness: .42, metalness: .22 });
    const fetMaterial = new THREE.MeshStandardMaterial({
      color: 0x78a587,
      roughness: .4,
      metalness: .25,
      emissive: 0x000000
    });
    const metalMaterial = new THREE.MeshStandardMaterial({ color: 0x77887c, roughness: .28, metalness: .64 });
    const rimMaterial = new THREE.MeshStandardMaterial({ color: 0x9db6a2, roughness: .38, metalness: .5 });
    const darkMaterial = new THREE.MeshStandardMaterial({ color: 0x526c61, roughness: .32, metalness: .68 });
    const copperMaterial = new THREE.MeshStandardMaterial({ color: 0xd69570, roughness: .38, metalness: .65 });
    const shaftMaterial = new THREE.MeshStandardMaterial({ color: 0xaeb9ae, roughness: .25, metalness: .8 });
    const motorBodyMaterial = new THREE.MeshStandardMaterial({
      color: 0xd8c6a2,
      roughness: .48,
      metalness: .12,
      transparent: true,
      opacity: .3,
      depthWrite: false
    });

    const addBox = (parent, size, material, position) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
      mesh.position.set(...position);
      parent.add(mesh);
      return mesh;
    };

    const addTextLabel = (parent, text, position, color = "#46554c") => {
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 128;
      const context = canvas.getContext("2d");
      context.fillStyle = "rgba(255,254,250,0.96)";
      context.fillRect(10, 10, 492, 108);
      context.fillStyle = color;
      context.font = "600 38px system-ui, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(text, 256, 64, 460);
      const texture = new THREE.CanvasTexture(canvas);
      texture.encoding = THREE.sRGBEncoding;
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
      sprite.position.set(...position);
      sprite.scale.set(1.45, .36, 1);
      parent.add(sprite);
      return sprite;
    };

    const rotor = new THREE.Group();
    const motor = new THREE.Group();
    assembly.add(motor);

    const axialCylinder = (parent, radiusTop, radiusBottom, length, material, x = 0, radialSegments = 48) => {
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radiusTop, radiusBottom, length, radialSegments), material);
      mesh.rotation.z = Math.PI / 2;
      mesh.position.x = x;
      parent.add(mesh);
      return mesh;
    };

    axialCylinder(motor, 1.02, 1.02, 2.2, motorBodyMaterial, 0);
    axialCylinder(motor, 1.04, 1.04, .11, rimMaterial, -1.08);
    axialCylinder(motor, 1.04, 1.04, .11, rimMaterial, 1.08);
    axialCylinder(motor, .79, .79, .13, darkMaterial, -1.16);
    axialCylinder(motor, .79, .79, .13, darkMaterial, 1.16);
    axialCylinder(motor, .55, .55, 1.65, darkMaterial, 0);
    motor.add(rotor);
    axialCylinder(rotor, .31, .31, 2.9, shaftMaterial, 0);
    axialCylinder(rotor, .61, .61, 1.05, copperMaterial, 0);
    axialCylinder(rotor, .67, .67, .16, rimMaterial, -.52);
    axialCylinder(rotor, .67, .67, .16, rimMaterial, .52);
    for (let index = 0; index < 4; index += 1) {
      const coil = new THREE.Mesh(new THREE.TorusGeometry(.56, .055, 10, 36), copperMaterial);
      coil.rotation.y = Math.PI / 2;
      coil.position.x = -.35 + index * .23;
      rotor.add(coil);
    }
    for (let index = 0; index < 8; index += 1) {
      const angle = (index / 8) * Math.PI * 2;
      const magnet = new THREE.Mesh(new THREE.BoxGeometry(.2, .14, .28), index % 2 ? rimMaterial : copperMaterial);
      magnet.position.set(0, Math.cos(angle) * .78, Math.sin(angle) * .78);
      magnet.rotation.x = angle;
      rotor.add(magnet);
    }
    for (let index = 0; index < 6; index += 1) {
      const blade = new THREE.Mesh(new THREE.BoxGeometry(.12, .68, .09), copperMaterial);
      const angle = (index / 6) * Math.PI * 2;
      blade.position.set(0, Math.cos(angle) * .34, Math.sin(angle) * .34);
      blade.rotation.x = angle;
      rotor.add(blade);
    }
    for (const x of [-1.34, 1.34]) {
      axialCylinder(rotor, .15, .15, .42, shaftMaterial, x, 32);
    }

    addBox(assembly, [9.3, .12, 3.7], boardMaterial, [0, -1.43, 0]);

    const source = new THREE.Group();
    source.position.set(-3.65, 0, .1);
    assembly.add(source);
    addBox(source, [1.05, 1.25, .9], sourceMaterial, [0, 0, 0]);
    addBox(source, [.72, .12, .68], rimMaterial, [0, .55, 0]);
    const postGeometry = new THREE.CylinderGeometry(.075, .075, .18, 20);
    [-.25, .25].forEach((x, index) => {
      const post = new THREE.Mesh(postGeometry, metalMaterial);
      post.position.set(x, .7, 0);
      source.add(post);
      addTextLabel(source, index === 0 ? "+" : "−", [x, .86, .05], index === 0 ? "#558664" : "#8d7060").scale.set(.25, .22, 1);
    });
    addTextLabel(source, "DC source", [0, -.87, .1]);

    const capacitor = new THREE.Group();
    capacitor.position.set(3.65, 0, .1);
    assembly.add(capacitor);
    const can = new THREE.Mesh(new THREE.CylinderGeometry(.48, .48, 1.35, 40), capacitorMaterial);
    capacitor.add(can);
    const capBand = new THREE.Mesh(new THREE.CylinderGeometry(.5, .5, .13, 40), rimMaterial);
    capBand.position.y = -.48;
    capacitor.add(capBand);
    const capTop = new THREE.Mesh(new THREE.CylinderGeometry(.1, .1, .22, 20), metalMaterial);
    capTop.position.y = .76;
    capacitor.add(capTop);
    addTextLabel(capacitor, "CAPACITOR", [0, -.88, .1], "#946b4f");

    const switchPositions = [
      [-1.72, .94, 1.0],
      [-1.72, -.94, 1.0],
      [1.72, .94, 1.0],
      [1.72, -.94, 1.0]
    ];
    const switches = switchPositions.map((position, index) => {
      const device = new THREE.Group();
      device.position.set(...position);
      assembly.add(device);
      const body = addBox(device, [.64, .48, .26], fetMaterial.clone(), [0, 0, 0]);
      addBox(device, [.28, .12, .06], metalMaterial, [0, .29, -.02]);
      for (const x of [-.2, 0, .2]) {
        const leg = new THREE.Mesh(new THREE.CylinderGeometry(.025, .025, .2, 10), metalMaterial);
        leg.position.set(x, -.33, 0);
        device.add(leg);
      }
      const led = new THREE.Mesh(
        new THREE.SphereGeometry(.07, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0xc5c8bb })
      );
      led.position.set(.22, .14, .16);
      device.add(led);
      addTextLabel(device, `Q${index + 1} · MOSFET`, [0, -.58, .08], "#46554c").scale.set(.96, .24, 1);
      return { body, led };
    });

    const makePath = (points, color) => {
      const curve = new THREE.CatmullRomCurve3(points.map((point) => new THREE.Vector3(...point)));
      const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .12 });
      const cable = new THREE.Mesh(new THREE.TubeGeometry(curve, 128, .045, 8, false), material);
      assembly.add(cable);
      const particleMaterial = new THREE.MeshBasicMaterial({ color });
      const particles = Array.from({ length: 5 }, () => {
        const particle = new THREE.Mesh(new THREE.SphereGeometry(.09, 12, 12), particleMaterial);
        particle.visible = false;
        assembly.add(particle);
        return particle;
      });
      return { curve, material, particles };
    };

    const motoringPath = makePath([
      [-3.1, .7, .22],
      [-2.25, .7, .82],
      [-1.72, .94, 1.14],
      [-.95, .38, 1.12],
      [0, .35, 1.13],
      [.92, -.38, 1.12],
      [1.72, -.94, 1.14],
      [2.4, -.74, .82],
      [1.72, -1.32, .92],
      [-1.72, -1.32, .92],
      [-2.4, -.74, .82],
      [-3.12, -.62, .22]
    ], 0x5ca879);

    const regenPath = makePath([
      [1.0, .45, 1.13],
      [1.72, .94, 1.14],
      [2.42, .78, .82],
      [3.65, .78, .36],
      [3.65, -.73, .36],
      [2.4, -.78, .82],
      [1.72, -.94, 1.14],
      [.85, -.42, 1.13],
      [0, -.4, 1.13],
      [-.95, .38, 1.12]
    ], 0xd68d67);

    const target = new THREE.Vector2();
    let dragging = false;
    let pointerX = 0;
    let pointerY = 0;
    renderer.domElement.addEventListener("pointerdown", (event) => {
      dragging = true;
      pointerX = event.clientX;
      pointerY = event.clientY;
      renderer.domElement.setPointerCapture(event.pointerId);
    });
    renderer.domElement.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      target.y += (event.clientX - pointerX) * .008;
      target.x += (event.clientY - pointerY) * .006;
      target.x = Math.max(-.8, Math.min(.8, target.x));
      pointerX = event.clientX;
      pointerY = event.clientY;
    });
    renderer.domElement.addEventListener("pointerup", () => { dragging = false; });
    renderer.domElement.addEventListener("wheel", (event) => {
      event.preventDefault();
      camera.position.multiplyScalar(event.deltaY > 0 ? 1.06 : .94);
      camera.position.clampLength(7, 18);
    }, { passive: false });

    state.motor = {
      renderer,
      camera,
      container,
      rotor,
      motor,
      assembly,
      target,
      switches,
      mode: null,
      paths: { motoring: motoringPath, regen: regenPath }
    };
    renderer.domElement.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      console.error("WebGL context lost; switching the powertrain twin to its 2D renderer.");
      setupMotorFallback(container);
    });
    if ("ResizeObserver" in window) {
      state.motor.resizeObserver = new ResizeObserver(resizeMotor);
      state.motor.resizeObserver.observe(container);
    }
    setDriveMode("coasting");
    resizeMotor();
    const clock = new THREE.Clock();
    let visualRpm = 0;
    let elapsed = 0;
    const animate = () => {
      window.requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), .05);
      elapsed += dt;
      const rpm = state.latest ? Math.max(0, Number(state.latest.rpm) || 0) : 0;
      const commandedRpm = state.driveControl.throttle > 0
        ? Math.max(30, state.driveControl.throttle * 900)
        : 0;
      const targetRpm = Math.max(rpm, commandedRpm);
      visualRpm += (targetRpm - visualRpm) * Math.min(1, dt * 2.5);
      rotor.rotation.x += visualRpm * (Math.PI * 2 / 60) * dt * .28;
      assembly.rotation.y += (target.y - assembly.rotation.y) * .08;
      assembly.rotation.z += (target.x - assembly.rotation.z) * .08;

      for (const [mode, path] of Object.entries(state.motor.paths)) {
        const active = state.motor.mode === mode;
        path.material.opacity += ((active ? .9 : .12) - path.material.opacity) * Math.min(1, dt * 3);
        path.particles.forEach((particle, index) => {
          particle.visible = active;
          if (active) {
            const progress = (elapsed * .22 + index / path.particles.length) % 1;
            particle.position.copy(path.curve.getPointAt(progress));
          }
        });
      }
      renderer.render(scene, camera);
    };
    animate();
  }

  window.addEventListener("popstate", navigate);
  window.addEventListener("hashchange", navigate);
  window.addEventListener("resize", () => {
    resizeMotor();
    drawCharts();
  });
  window.setInterval(() => {
    if (window.location.pathname === "/admin") refreshAdmin();
  }, 5000);
  navigate();
  setupControls();
  setupBrakingCalculator();
  setupOptimizer();
  setupSafety();
  setupMotor();
  refreshHealth();
  connectSocket();
})();
