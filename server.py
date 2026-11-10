#!/usr/bin/env python3
"""
===============================================================================
PROJECT: Real-Time Telemetry, Edge Analytics, and 3D Digital Twin for a
         Regenerative DC Drive Powertrain
MODULE 3: FASTAPI & WEBSOCKET TELEMETRY BACKEND
SCRIPT:   server.py
===============================================================================

Domain: Electrical & Electronics Engineering (EEE) Powertrain Edge Analytics
Hardware Target:
  - Custom H-Bridge Motor Driver (IRF9540N / IRFZ44N / 1000 uF / 0.05 Ohm shunt)
  - Microcontroller: ESP32 / Arduino Wi-Fi Telemetry Node
  - DC Bus: 12V Nominal Rail (11.4V to 12.6V)
  - 15 kHz Dynamic Braking Low-Side PWM Boost Regeneration

Backend Architecture:
  1. FastAPI Application with CORS middleware & static file hosting.
  2. Lifespan model loader: loads scaler.joblib, energy_model.joblib,
     stopping_model.joblib, and safety_classifier.joblib on startup.
  3. WebSocket Endpoint (/ws/telemetry):
       - Mode A (Physical Hardware): Ingests ESP32 JSON telemetry packets,
         evaluates real-time ML inferences, and broadcasts to dashboard clients.
       - Mode B (Internal 60 Hz Physics Loop): If no hardware packets are
         received, runs a 60 Hz electromechanical simulation loop responding to
         real-time throttle and brake controls.
  4. REST Endpoint (/api/predict): Ingests feature parameters, evaluates all
     three ML models, sweeps duty cycle, and returns the optimal braking duty.
  5. REST Endpoint (/api/status): System health, mode, and telemetry snapshot.
"""

import os
import time
import json
import asyncio
import argparse
import secrets
from typing import Dict, Any, List, Optional, Set

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_STARTED_AT = time.monotonic()
admin_basic_auth = HTTPBasic(auto_error=False)
ADMIN_SESSION_COOKIE = "regen_admin_session"
ADMIN_SESSION_TTL_SECONDS = 8 * 60 * 60
ADMIN_SESSIONS: Dict[str, tuple[str, float]] = {}
MODEL_FEATURES = [
    "initial_rpm",
    "pwm_brake_duty",
    "bus_voltage_init",
    "motor_temp_c",
    "peak_current_a",
    "peak_kickback_voltage_v",
    "mean_current_ripple",
]


def configured_admin_credentials() -> tuple[Optional[str], Optional[str]]:
    username = os.environ.get("ADMIN_USERNAME")
    password = os.environ.get("ADMIN_PASSWORD")
    return username, password


def valid_admin_credentials(username: str, password: str) -> bool:
    expected_username, expected_password = configured_admin_credentials()
    return bool(
        expected_username
        and expected_password
        and secrets.compare_digest(username.encode("utf-8"), expected_username.encode("utf-8"))
        and secrets.compare_digest(password.encode("utf-8"), expected_password.encode("utf-8"))
    )


def authenticated_admin(
    request: Request,
    credentials: Optional[HTTPBasicCredentials],
) -> Optional[str]:
    username, password = configured_admin_credentials()
    if not username or not password:
        return None

    token = request.cookies.get(ADMIN_SESSION_COOKIE)
    if token:
        session = ADMIN_SESSIONS.get(token)
        if session and session[1] > time.monotonic() and secrets.compare_digest(session[0], username):
            return username
        ADMIN_SESSIONS.pop(token, None)

    if credentials and valid_admin_credentials(credentials.username, credentials.password):
        return credentials.username
    return None


def require_admin(
    request: Request,
    credentials: Optional[HTTPBasicCredentials] = Depends(admin_basic_auth),
) -> str:
    """Require configured credentials or a valid, unexpired admin session."""
    username, password = configured_admin_credentials()
    if not username or not password:
        raise HTTPException(
            status_code=503,
            detail="Admin authentication is not configured. Set ADMIN_USERNAME and ADMIN_PASSWORD before starting the server.",
        )
    authenticated_username = authenticated_admin(request, credentials)
    if authenticated_username is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials.",
        )
    return authenticated_username


class AdminLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


def admin_login_page(message: Optional[str] = None) -> str:
    """Render a small sign-in page so login works in browsers without Basic prompts."""
    status = (
        f'<p class="message setup">{message}</p>'
        if message
        else '<p class="message" id="login-message" role="alert" aria-live="polite"></p>'
    )
    form = "" if message else """
      <form id="admin-login-form">
        <label for="username">Username</label>
        <input id="username" name="username" autocomplete="username" required>
        <label for="password">Password</label>
        <input id="password" name="password" type="password" autocomplete="current-password" required>
        <button type="submit">Sign in</button>
      </form>
      <script>
        document.querySelector("#admin-login-form").addEventListener("submit", async (event) => {
          event.preventDefault();
          const button = event.currentTarget.querySelector("button");
          const message = document.querySelector("#login-message");
          button.disabled = true;
          message.textContent = "Signing in…";
          try {
            const response = await fetch("/api/admin/login", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              credentials: "same-origin",
              body: JSON.stringify({
                username: document.querySelector("#username").value,
                password: document.querySelector("#password").value
              })
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || "Sign-in failed.");
            window.location.replace("/admin");
          } catch (error) {
            message.textContent = error instanceof Error ? error.message : "Unable to sign in.";
            button.disabled = false;
          }
        });
      </script>
    """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Admin sign in · Regen Studio</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ min-height: 100vh; display: grid; place-items: center; margin: 0; padding: 24px; background: #f5f4ed; color: #35463c; font: 15px/1.5 system-ui, sans-serif; }}
    main {{ width: min(100%, 390px); padding: 30px; border: 1px solid #e6e5dc; border-radius: 20px; background: #fffefa; box-shadow: 0 16px 48px #45574712; }}
    .mark {{ display: grid; place-items: center; width: 42px; height: 42px; border-radius: 14px; background: #dcecdf; color: #496d54; font-weight: 800; }}
    h1 {{ margin: 20px 0 4px; font-size: 24px; }}
    .intro {{ margin: 0 0 23px; color: #7d887f; font-size: 13px; }}
    label {{ display: block; margin: 15px 0 6px; font-size: 12px; font-weight: 650; }}
    input {{ width: 100%; padding: 12px; border: 1px solid #dedfd5; border-radius: 10px; background: #fff; color: inherit; font: inherit; }}
    input:focus {{ outline: 2px solid #94b99b; outline-offset: 1px; }}
    button {{ width: 100%; margin-top: 21px; padding: 12px; border: 0; border-radius: 10px; background: #7fa88a; color: white; font: inherit; font-weight: 700; cursor: pointer; }}
    button:disabled {{ opacity: .6; cursor: wait; }}
    .message {{ min-height: 20px; margin: 12px 0 0; color: #a35451; font-size: 12px; }}
    .message.setup {{ color: #8a6944; }}
    .security {{ margin: 12px 0 0; color: #8b948b; font-size: 11px; }}
  </style>
</head>
<body>
  <main>
    <div class="mark" aria-hidden="true">R</div>
    <h1>Admin sign in</h1>
    <p class="intro">Regen Studio · Powertrain lab</p>
    {status}
    {form}
    <p class="security">Authorized users only. Your sign-in creates a private eight-hour session.</p>
  </main>
</body>
</html>"""

# =============================================================================
# 1. APPLICATION & MODEL REGISTRY
# =============================================================================

app = FastAPI(
    title="Regenerative DC Drive Telemetry & Digital Twin Backend",
    version="1.0.0",
    description="High-performance asynchronous telemetry ingestion and edge analytics engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelRegistry:
    """Holds loaded machine learning models and scaler instances."""
    scaler = None
    energy_model = None
    stopping_model = None
    safety_classifier = None
    loaded: bool = False

    @classmethod
    def load_artifacts(cls) -> bool:
        """Loads the four joblib artifacts located beside this module."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts = {
            "scaler": os.path.join(base_dir, "scaler.joblib"),
            "energy_model": os.path.join(base_dir, "energy_model.joblib"),
            "stopping_model": os.path.join(base_dir, "stopping_model.joblib"),
            "safety_classifier": os.path.join(base_dir, "safety_classifier.joblib"),
        }
        try:
            for key, filepath in artifacts.items():
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Missing artifact: {filepath}")
                setattr(cls, key, joblib.load(filepath))
                print(f"[+] Successfully loaded {key} ({os.path.getsize(filepath)} bytes)")
            cls.loaded = True
            print("[+] All 4 ML pipeline artifacts loaded successfully into memory.")
            return True
        except Exception as e:
            print(f"[!] Error loading model artifacts: {e}")
            cls.loaded = False
            return False


# =============================================================================
# 2. ELECTROMECHANICAL PHYSICS SIMULATION ENGINE (60 Hz INTERNAL LOOP)
# =============================================================================

class PowertrainPhysicsEngine:
    """
    First-principles 60 Hz real-time simulator for DC drive powertrain.
    Models motoring, coasting, and 15 kHz boost regenerative braking.
    """
    def __init__(self):
        # Motor parameters
        self.Ra_25 = 1.8           # Ohm
        self.alpha_Cu = 0.00393    # 1/deg C
        self.La = 2.5e-3           # H
        self.Kt = 0.045            # N*m/A
        self.Ke = 0.045            # V/(rad/s)
        self.J = 0.0008            # kg*m^2
        self.B = 0.0001            # N*m*s/rad
        self.tau_friction = 0.035  # N*m

        # Hardware circuit
        self.R_bat = 0.08          # Ohm
        self.V_bat_nominal = 12.0  # V
        self.V_diode = 1.15        # V

        # Dynamic State Variables
        self.omega = 0.0           # rad/s
        self.v_bat = 12.0          # V
        self.i_shunt = 0.0         # A
        self.temp_c = 35.0         # deg C
        self.throttle = 0.0        # 0.0 to 1.0
        self.brake_duty = 0.0      # 0.0 to 1.0
        self.peak_current = 0.0    # A
        self.peak_kickback = 12.0  # V
        self.current_ripple = 0.0  # A

        self.rng = np.random.default_rng(42)

    @property
    def rpm(self) -> float:
        return float(self.omega * 60.0 / (2.0 * np.pi))

    def set_controls(self, throttle: float, brake_duty: float) -> None:
        self.throttle = float(np.clip(throttle, 0.0, 1.0))
        self.brake_duty = float(np.clip(brake_duty, 0.0, 0.95))

    def step(self, dt: float = 0.01667) -> Dict[str, Any]:
        """Advances physics state by dt seconds (60 Hz update)."""
        Ra = self.Ra_25 * (1.0 + self.alpha_Cu * (self.temp_c - 25.0))
        eb = self.Ke * self.omega

        # --- Motoring Mode (Throttle Active, Brake Inactive) ---
        if self.throttle > 0.01 and self.brake_duty <= 0.05:
            v_applied = self.v_bat * self.throttle
            di_dt = (v_applied - eb - (Ra + 0.05) * self.i_shunt) / self.La
            self.i_shunt = float(np.clip(self.i_shunt + di_dt * dt, 0.0, 15.0))

            torque_motor = self.Kt * self.i_shunt
            torque_load = self.B * self.omega + self.tau_friction
            domega_dt = (torque_motor - torque_load) / self.J
            self.omega = float(max(0.0, self.omega + domega_dt * dt))

            self.v_bat = float(max(11.0, self.V_bat_nominal - self.R_bat * self.i_shunt))
            self.peak_current = float(self.i_shunt)
            self.peak_kickback = float(self.v_bat)
            self.current_ripple = float(0.02 * self.throttle)

        # --- Boost Regenerative Braking Mode (Brake Active) ---
        elif self.brake_duty > 0.05:
            d = self.brake_duty
            Ron = Ra + (0.0175 * 2.0) + 0.05
            Roff = Ra + 0.0175 + 0.05

            Iss_on = eb / Ron
            Iss_off = (eb - (self.v_bat + self.V_diode)) / Roff

            i_target = max(0.0, d * Iss_on + (1.0 - d) * Iss_off)
            self.i_shunt = float(0.85 * self.i_shunt + 0.15 * i_target)

            if self.i_shunt > self.peak_current:
                self.peak_current = float(self.i_shunt)

            t_on = d / 15000.0
            self.current_ripple = float(min(1.5, (eb / self.La) * t_on))
            self.peak_kickback = float(self.v_bat + self.V_diode + (0.16 * self.i_shunt))

            i_regen = float(max(0.0, (1.0 - d) * self.i_shunt))
            self.v_bat = float(self.V_bat_nominal + self.R_bat * i_regen)

            torque_brake = self.Kt * self.i_shunt
            torque_load = self.B * self.omega + self.tau_friction
            domega_dt = (torque_brake + torque_load) / self.J
            self.omega = float(max(0.0, self.omega - domega_dt * dt))

        # --- Coasting Mode (Zero Throttle & Zero Brake) ---
        else:
            self.i_shunt = float(max(0.0, self.i_shunt * 0.90))
            torque_load = self.B * self.omega + self.tau_friction
            domega_dt = torque_load / self.J
            self.omega = float(max(0.0, self.omega - domega_dt * dt))
            self.v_bat = float(self.V_bat_nominal)
            self.current_ripple = 0.005
            self.peak_kickback = float(self.v_bat)

        # Thermal dissipation
        p_copper = (self.i_shunt ** 2) * Ra
        self.temp_c = float(np.clip(self.temp_c + (p_copper * 0.002 - (self.temp_c - 25.0) * 0.001) * dt, 20.0, 95.0))

        # Sensor noise injection
        noisy_i = float(max(0.0, self.i_shunt + self.rng.normal(0.0, 0.025)))
        noisy_v = float(max(10.0, self.v_bat + self.rng.normal(0.0, 0.035)))

        return {
            "rpm": round(self.rpm, 1),
            "v_bat": round(noisy_v, 2),
            "i_shunt": round(noisy_i, 3),
            "temp": round(self.temp_c, 1),
            "duty": round(self.brake_duty, 3),
            "throttle": round(self.throttle, 2),
            "peak_current_a": round(self.peak_current, 3),
            "peak_kickback_voltage_v": round(self.peak_kickback, 2),
            "mean_current_ripple": round(self.current_ripple, 4)
        }


# =============================================================================
# 3. WEBSOCKET CONNECTION & TELEMETRY MANAGER
# =============================================================================

class ConnectionManager:
    """Manages active WebSocket connections and hardware vs. simulator modes."""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.mode: str = "simulation"  # "hardware" or "simulation"
        self.last_hardware_time: float = 0.0
        self.physics_engine = PowertrainPhysicsEngine()
        self.latest_telemetry: Dict[str, Any] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        dead = []
        payload = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.active_connections.discard(d)

    def evaluate_ml(self, rpm: float, duty: float, v_bat: float, temp: float,
                    i_peak: float, v_kick: float, ripple: float) -> Dict[str, Any]:
        """Runs trained ML models on telemetry feature vector."""
        if not ModelRegistry.loaded:
            return {
                "energy_recovered_joules": 0.0,
                "stopping_time_ms": 0.0,
                "safety_class": 0,
                "safety_label": "Models Not Loaded"
            }

        try:
            raw_features = np.array([[rpm, duty, v_bat, temp, i_peak, v_kick, ripple]], dtype=np.float64)
            scaled = ModelRegistry.scaler.transform(raw_features)

            e_rec = float(ModelRegistry.energy_model.predict(scaled)[0])
            s_time = float(ModelRegistry.stopping_model.predict(scaled)[0])
            s_class = int(ModelRegistry.safety_classifier.predict(scaled)[0])

            labels = {
                0: "Class 0 (Inefficient)",
                1: "Class 1 (Optimal Safe Regen)",
                2: "Class 2 (Overvoltage/Thermal Hazard)"
            }

            return {
                "energy_recovered_joules": round(e_rec, 4),
                "stopping_time_ms": round(s_time, 1),
                "safety_class": s_class,
                "safety_label": labels.get(s_class, f"Class {s_class}")
            }
        except Exception as e:
            return {
                "energy_recovered_joules": 0.0,
                "stopping_time_ms": 0.0,
                "safety_class": 0,
                "safety_label": f"Error: {e}"
            }

manager = ConnectionManager()


# =============================================================================
# 4. BACKGROUND 60 Hz ASYNCHRONOUS PHYSICS LOOP
# =============================================================================

async def physics_loop():
    """60 Hz async simulation loop broadcasting live state when in Mode B."""
    print("[*] Starting 60 Hz electromechanical physics background loop...")
    dt = 1.0 / 60.0
    counter = 0

    while True:
        await asyncio.sleep(dt)
        now = time.time()

        if now - manager.last_hardware_time > 1.5:
            manager.mode = "simulation"

        if manager.mode == "simulation":
            state = manager.physics_engine.step(dt)
            counter += 1

            if counter % 4 == 0 or "ml_predictions" not in manager.latest_telemetry:
                ml_res = manager.evaluate_ml(
                    state["rpm"], state["duty"], state["v_bat"], state["temp"],
                    state["peak_current_a"], state["peak_kickback_voltage_v"], state["mean_current_ripple"]
                )
                state["ml_predictions"] = ml_res

            state["timestamp"] = round(now, 3)
            state["mode"] = manager.mode
            manager.latest_telemetry = state

            if manager.active_connections:
                await manager.broadcast(state)


@app.on_event("startup")
async def startup_event():
    ModelRegistry.load_artifacts()
    asyncio.create_task(physics_loop())


# =============================================================================
# 5. REST & WEBSOCKET ENDPOINTS
# =============================================================================

class PredictRequest(BaseModel):
    initial_rpm: float = Field(..., ge=100.0, le=4500.0, description="Entry rotor shaft speed in RPM")
    pwm_brake_duty: Optional[float] = Field(0.50, ge=0.0, le=1.0, description="PWM braking duty cycle")
    bus_voltage_init: Optional[float] = Field(12.0, ge=9.0, le=16.0, description="DC bus battery voltage in V")
    motor_temp_c: Optional[float] = Field(35.0, ge=15.0, le=110.0, description="Motor core temperature in deg C")
    peak_current_a: Optional[float] = None
    peak_kickback_voltage_v: Optional[float] = None
    mean_current_ripple: Optional[float] = None


@app.post("/api/predict")
async def predict_telemetry(req: PredictRequest):
    """
    Evaluates ML predictions and sweeps duty cycles to compute the optimal
    regenerative braking duty cycle.
    """
    if not ModelRegistry.loaded:
        raise HTTPException(status_code=503, detail="Machine learning models are not loaded.")

    rpm = req.initial_rpm
    v_bus = req.bus_voltage_init if req.bus_voltage_init is not None else 12.0
    temp = req.motor_temp_c if req.motor_temp_c is not None else 35.0
    duty = req.pwm_brake_duty if req.pwm_brake_duty is not None else 0.50

    eb = 0.045 * (rpm * 2.0 * np.pi / 60.0)
    i_peak = req.peak_current_a if req.peak_current_a is not None else float(max(0.05, (eb / 1.9) * (duty ** 0.8)))
    v_kick = req.peak_kickback_voltage_v if req.peak_kickback_voltage_v is not None else float(v_bus + 1.15 + 0.16 * i_peak)
    ripple = req.mean_current_ripple if req.mean_current_ripple is not None else float(0.01 + 0.05 * duty * (1.0 - duty))

    ml_current = manager.evaluate_ml(rpm, duty, v_bus, temp, i_peak, v_kick, ripple)

    # Sweep duty cycles [0.15 to 0.85] to identify optimal regenerative braking
    sweep_duties = np.linspace(0.15, 0.85, 15)
    duty_sweep_results = []
    best_duty = duty
    max_safe_energy = -999.0

    for d in sweep_duties:
        d_val = float(np.round(d, 3))
        est_ipeak = float(max(0.05, (eb / 1.9) * (d_val ** 0.8)))
        est_vkick = float(v_bus + 1.15 + 0.16 * est_ipeak)
        est_ripple = float(0.01 + 0.05 * d_val * (1.0 - d_val))

        pred = manager.evaluate_ml(rpm, d_val, v_bus, temp, est_ipeak, est_vkick, est_ripple)
        item = {
            "duty": d_val,
            "energy_recovered_j": pred["energy_recovered_joules"],
            "stopping_time_ms": pred["stopping_time_ms"],
            "safety_class": pred["safety_class"],
            "safety_label": pred["safety_label"]
        }
        duty_sweep_results.append(item)

        if pred["safety_class"] == 1 and pred["energy_recovered_joules"] > max_safe_energy:
            max_safe_energy = pred["energy_recovered_joules"]
            best_duty = d_val

    if max_safe_energy == -999.0:
        for item in duty_sweep_results:
            if item["safety_class"] == 0 and item["energy_recovered_j"] > max_safe_energy:
                max_safe_energy = item["energy_recovered_j"]
                best_duty = item["duty"]

    return {
        "status": "success",
        "input_features": {
            "initial_rpm": rpm,
            "pwm_brake_duty": duty,
            "bus_voltage_init": v_bus,
            "motor_temp_c": temp,
            "peak_current_a": round(i_peak, 3),
            "peak_kickback_voltage_v": round(v_kick, 2),
            "mean_current_ripple": round(ripple, 4)
        },
        "current_prediction": ml_current,
        "optimization": {
            "optimal_brake_duty": best_duty,
            "expected_energy_recovered_j": max_safe_energy if max_safe_energy != -999.0 else ml_current["energy_recovered_joules"],
            "duty_sweep": duty_sweep_results
        }
    }


@app.get("/api/status")
async def server_status():
    """Returns server operational health and latest telemetry state."""
    return {
        "status": "online",
        "models_loaded": ModelRegistry.loaded,
        "mode": manager.mode,
        "active_clients": len(manager.active_connections),
        "latest_telemetry": manager.latest_telemetry
    }


def describe_estimator(
    role: str,
    target: str,
    where_used: str,
    why: str,
    parameters: List[str],
) -> Dict[str, Any]:
    model = getattr(ModelRegistry, role, None)
    if model is None:
        return {
            "role": role,
            "target": target,
            "where_used": where_used,
            "why": why,
            "loaded": False,
            "algorithm": "Unavailable",
            "parameters": {},
            "feature_importances": [],
        }

    get_params = getattr(model, "get_params", None)
    model_params = get_params(deep=False) if callable(get_params) else {}
    fitted_importances = getattr(model, "feature_importances_", None)
    importance_rows = []
    if fitted_importances is not None and len(fitted_importances) == len(MODEL_FEATURES):
        importance_rows = [
            {"feature": MODEL_FEATURES[index], "importance": float(fitted_importances[index])}
            for index in np.argsort(fitted_importances)[::-1]
        ]

    return {
        "role": role,
        "target": target,
        "where_used": where_used,
        "why": why,
        "loaded": True,
        "algorithm": type(model).__name__,
        "parameters": {key: model_params[key] for key in parameters if key in model_params},
        "feature_importances": importance_rows,
    }


@app.get("/api/admin/overview")
async def admin_overview(
    response: Response,
    _admin_username: str = Depends(require_admin),
):
    """Returns authenticated runtime and model-method details for the admin portal."""
    response.headers["Cache-Control"] = "no-store"
    model_files = {
        "scaler": "scaler.joblib",
        "energy_model": "energy_model.joblib",
        "stopping_model": "stopping_model.joblib",
        "safety_classifier": "safety_classifier.joblib",
    }
    artifacts = []
    for key, filename in model_files.items():
        filepath = os.path.join(BASE_DIR, filename)
        artifacts.append({
            "name": filename,
            "role": key,
            "loaded": bool(ModelRegistry.loaded and getattr(ModelRegistry, key, None) is not None),
            "present": os.path.isfile(filepath),
            "size_bytes": os.path.getsize(filepath) if os.path.isfile(filepath) else None,
        })

    engine = manager.physics_engine
    return {
        "status": "online",
        "uptime_seconds": round(time.monotonic() - SERVER_STARTED_AT, 1),
        "mode": manager.mode,
        "active_clients": len(manager.active_connections),
        "models_loaded": ModelRegistry.loaded,
        "model_artifacts": artifacts,
        "analytics": {
            "training_data": {
                "source": "Training pipeline default: motor_regen_dataset.csv; the serialized model artifacts do not record their dataset path.",
                "provenance": "The project generator creates simulation-derived examples; this is not measured vehicle-fleet data.",
                "split": "80% train / 20% test, random_state=42, stratified by safety class.",
                "evaluation": "Regression: R², MAE, RMSE. Safety classifier: confusion matrix and per-class precision/recall/F1.",
                "metrics_note": "Training code calculates and prints these holdout metrics, but does not persist their values with the model artifacts; no live accuracy claim is made here.",
                "features": MODEL_FEATURES,
            },
            "preprocessing": {
                "algorithm": type(ModelRegistry.scaler).__name__ if ModelRegistry.scaler is not None else "Unavailable",
                "where_used": "Applied to the seven input features immediately before each of the three model predictions.",
                "why": "Keeps one consistent preprocessing contract between training and runtime inference. The selected tree ensembles are generally scale-invariant, so scaling is not claimed to improve their split quality.",
            },
            "estimators": [
                describe_estimator(
                    "energy_model",
                    "energy_recovered_joules (continuous)",
                    "Live telemetry inference and every regen-optimizer duty candidate.",
                    "Random Forest averages many bootstrapped decision trees, capturing nonlinear feature interactions while reducing the variance of a single tree for a continuous energy target.",
                    ["n_estimators", "max_depth", "random_state"],
                ),
                describe_estimator(
                    "stopping_model",
                    "stopping_time_ms (continuous)",
                    "Live telemetry inference and every regen-optimizer duty candidate.",
                    "Gradient Boosting adds shallow trees sequentially to correct prior residual errors; this fits a nonlinear continuous stopping-time target with a controlled learning rate.",
                    ["n_estimators", "max_depth", "learning_rate", "random_state"],
                ),
                describe_estimator(
                    "safety_classifier",
                    "safety_class (0 inefficient, 1 optimal-safe, 2 hazard)",
                    "Live telemetry classification and the safety constraint in each optimizer candidate.",
                    "A Random Forest classifier averages class-probability estimates across bootstrapped decision trees to distinguish the three nonlinear operating regimes; the predicted class is advisory, not a hardware interlock.",
                    ["n_estimators", "max_depth", "random_state"],
                ),
            ],
            "optimizer": {
                "algorithm": "Bounded exhaustive grid search (15-point duty-cycle sweep)",
                "where_used": "POST /api/predict, used by the Regen optimizer page.",
                "search_space": "15 evenly spaced PWM brake duties from 0.15 through 0.85, inclusive.",
                "objective": "Maximize predicted recovered joules among candidates predicted as safety class 1; if none qualify, fall back to the highest-energy class 0 candidate. If the sweep contains neither class, the code keeps the requested duty and current prediction.",
                "why": "Duty is one bounded scalar and only 15 candidates are evaluated, so a transparent deterministic sweep is simple and auditable. This is a discrete recommendation, not a continuous global optimum.",
            },
            "not_used": "No KNN, clustering, standalone decision-tree model, linear regression, or neural network is used in the current prediction/optimization path. Decision trees are the base learners inside the two Random Forests and Gradient Boosting model.",
        },
        "rates": {
            "simulation_hz": 60,
            "hardware_telemetry_hz": 50,
            "dashboard_readout_interval_ms": 1000,
        },
        "controls": {
            "throttle": round(engine.throttle, 3),
            "brake_duty": round(engine.brake_duty, 3),
        },
        "simulation_parameters": {
            "armature_resistance_ohms_at_25c": engine.Ra_25,
            "copper_temperature_coefficient_per_c": engine.alpha_Cu,
            "armature_inductance_h": engine.La,
            "torque_constant_nm_per_a": engine.Kt,
            "back_emf_constant_v_per_rad_s": engine.Ke,
            "rotor_inertia_kg_m2": engine.J,
            "viscous_friction_nm_s_per_rad": engine.B,
            "friction_torque_nm": engine.tau_friction,
            "battery_nominal_voltage_v": engine.V_bat_nominal,
            "battery_internal_resistance_ohms": engine.R_bat,
            "diode_drop_v": engine.V_diode,
        },
        "latest_telemetry": manager.latest_telemetry,
    }


@app.post("/api/admin/login")
async def admin_login(
    payload: AdminLoginRequest,
    request: Request,
    response: Response,
):
    """Create an eight-hour, HttpOnly admin session after checking credentials."""
    username, password = configured_admin_credentials()
    if not username or not password:
        raise HTTPException(
            status_code=503,
            detail="Admin authentication is not configured. Set ADMIN_USERNAME and ADMIN_PASSWORD before starting the server.",
        )
    if not valid_admin_credentials(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="Username or password is incorrect.")

    now = time.monotonic()
    expired_tokens = [token for token, (_, expiry) in ADMIN_SESSIONS.items() if expiry <= now]
    for token in expired_tokens:
        ADMIN_SESSIONS.pop(token, None)

    token = secrets.token_urlsafe(32)
    ADMIN_SESSIONS[token] = (username, now + ADMIN_SESSION_TTL_SECONDS)
    response.set_cookie(
        key=ADMIN_SESSION_COOKIE,
        value=token,
        max_age=ADMIN_SESSION_TTL_SECONDS,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return {"authenticated": True}


@app.post("/api/admin/logout")
async def admin_logout(
    request: Request,
    response: Response,
):
    token = request.cookies.get(ADMIN_SESSION_COOKIE)
    if token:
        ADMIN_SESSIONS.pop(token, None)
    response.delete_cookie(
        key=ADMIN_SESSION_COOKIE,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return {"authenticated": False}


@app.get("/admin", response_class=HTMLResponse)
async def serve_admin(
    request: Request,
    credentials: Optional[HTTPBasicCredentials] = Depends(admin_basic_auth),
):
    """Serve the diagnostics page after login, or a credential form otherwise."""
    username, password = configured_admin_credentials()
    if not username or not password:
        response = HTMLResponse(
            content=admin_login_page(
                "Admin login is not configured. Set ADMIN_USERNAME and ADMIN_PASSWORD in the server environment, then restart it."
            ),
            status_code=503,
        )
    elif authenticated_admin(request, credentials) is None:
        response = HTMLResponse(content=admin_login_page(), status_code=200)
    else:
        response = await serve_index()
    response.headers["Cache-Control"] = "no-store"
    return response


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    Bidirectional WebSocket telemetry endpoint:
      - Ingests hardware packets (Mode A: ESP32)
      - Receives dashboard control commands (throttle, brake duty)
      - Broadcasts synchronized 60 Hz telemetry and ML predictions
    """
    await manager.connect(websocket)
    try:
        if manager.latest_telemetry:
            await websocket.send_text(json.dumps(manager.latest_telemetry))

        while True:
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "")

            # Mode A: Hardware Ingestion Packet from ESP32
            if msg_type == "telemetry" or "rpm" in data:
                manager.mode = "hardware"
                manager.last_hardware_time = time.time()

                rpm = float(data.get("rpm", 0.0))
                v_bat = float(data.get("v_bat", 12.0))
                i_shunt = float(data.get("i_shunt", 0.0))
                temp = float(data.get("temp", 30.0))
                duty = float(data.get("duty", 0.0))

                i_peak = float(data.get("peak_current_a", max(i_shunt, 0.05)))
                v_kick = float(data.get("peak_kickback_voltage_v", v_bat + 1.15))
                ripple = float(data.get("mean_current_ripple", 0.02))

                ml_res = manager.evaluate_ml(rpm, duty, v_bat, temp, i_peak, v_kick, ripple)

                enriched_packet = {
                    "timestamp": round(time.time(), 3),
                    "mode": "hardware",
                    "rpm": round(rpm, 1),
                    "v_bat": round(v_bat, 2),
                    "i_shunt": round(i_shunt, 3),
                    "temp": round(temp, 1),
                    "duty": round(duty, 3),
                    "peak_current_a": round(i_peak, 3),
                    "peak_kickback_voltage_v": round(v_kick, 2),
                    "mean_current_ripple": round(ripple, 4),
                    "ml_predictions": ml_res
                }
                manager.latest_telemetry = enriched_packet
                await manager.broadcast(enriched_packet)

            # Dashboard Control Inputs (Sliders)
            elif msg_type == "control":
                throttle = data.get("throttle", manager.physics_engine.throttle)
                brake_duty = data.get("brake_duty", manager.physics_engine.brake_duty)
                manager.physics_engine.set_controls(throttle, brake_duty)

            # Mode Override Command
            elif msg_type == "mode":
                requested_mode = data.get("mode", "simulation")
                if requested_mode in ["hardware", "simulation"]:
                    manager.mode = requested_mode

            # Simulation Reset
            elif msg_type == "reset":
                manager.physics_engine = PowertrainPhysicsEngine()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# =============================================================================
# 6. STATIC FILES & FRONTEND ROUTING
# =============================================================================

STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    for candidate in [
        os.path.join(BASE_DIR, "index.html"),
        os.path.join(STATIC_DIR, "index.html"),
        os.path.join(BASE_DIR, "artifacts/file_generation/ttl=63d/output/index.html"),
    ]:
        if os.path.exists(candidate):
            return FileResponse(candidate)

    return HTMLResponse(content="""<!DOCTYPE html>
<html>
<head>
    <title>Regenerative Powertrain Digital Twin</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #e2e8f0; margin: 0; padding: 24px; }
        .card { background: #1a2234; border: 1px solid #2d3748; border-radius: 8px; padding: 20px; max-width: 600px; margin: 40px auto; }
        h2 { color: #38bdf8; margin-top: 0; }
        .status { color: #4ade80; font-weight: bold; }
        code { background: #0f172a; padding: 2px 6px; border-radius: 4px; color: #f472b6; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Regenerative Powertrain Edge Analytics Server</h2>
        <p>Status: <span class="status">ONLINE (60 Hz Engine Active)</span></p>
        <p>WebSocket Telemetry: <code>/ws/telemetry</code></p>
        <p>REST Inference API: <code>/api/predict</code></p>
        <p>System Health API: <code>/api/status</code></p>
    </div>
</body>
</html>""")

@app.get("/digital_twin_app.js")
async def serve_js():
    for candidate in [
        os.path.join(BASE_DIR, "digital_twin_app.js"),
        os.path.join(STATIC_DIR, "digital_twin_app.js"),
        os.path.join(BASE_DIR, "artifacts/file_generation/ttl=63d/output/digital_twin_app.js"),
    ]:
        if os.path.exists(candidate):
            return FileResponse(candidate, media_type="application/javascript")
    return JSONResponse(status_code=404, content={"error": "digital_twin_app.js not yet deployed"})


@app.get("/{page_name}", response_class=HTMLResponse)
async def serve_frontend_page(page_name: str):
    if page_name not in {"braking", "optimizer", "analytics", "safety"}:
        raise HTTPException(status_code=404, detail="Page not found.")
    return await serve_index()


# =============================================================================
# 7. CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="FastAPI & WebSocket Telemetry Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host address")
    parser.add_argument("--port", type=int, default=8000, help="Listening TCP port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run("server.py:app", host=args.host, port=args.port, reload=args.reload)