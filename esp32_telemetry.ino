/*
================================================================================
MODULE 4: EMBEDDED TELEMETRY & POWERTRAIN CONTROL FIRMWARE
Project: Real-Time Telemetry, Edge Analytics, and 3D Digital Twin
File:    esp32_telemetry.ino
Hardware Architecture:
  - Microcontroller: ESP32 DevKit V1 (Dual-Core 240 MHz)
  - Gate Drivers: BC547 NPN Inverting Gate Shifters
  - Power Switches: IRF9540N (P-Channel High-Side) & IRFZ44N (N-Channel Low-Side)
  - Telemetry: INA219 High-Side/Bidirectional Shunt & Resistor Divider
================================================================================
*/

#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_INA219.h>

// =============================================================================
// PIN DEFINITIONS & HARDWARE MAPPING
// =============================================================================
// Gate Control Outputs (Driven through BC547 Level Shifters)
#define PIN_PWM_HIGH_SIDE_A   25  // Drives High-Side P-MOSFET Phase A
#define PIN_PWM_LOW_SIDE_A    26  // Drives Low-Side N-MOSFET Phase A
#define PIN_PWM_HIGH_SIDE_B   27  // Drives High-Side P-MOSFET Phase B
#define PIN_PWM_LOW_SIDE_B    14  // Drives Low-Side N-MOSFET Phase B

// Analog Sensory Telemetry Pins
#define PIN_BUS_VOLTAGE_ADC   34  // Resistor Divider (100k / 10k) to 12V DC Rail
#define PIN_SHUNT_CURRENT_ADC 35  // Fallback Analog Current Shunt Pin (0.05 Ohm)
#define PIN_NTC_TEMP_ADC      32  // 10k NTC Thermistor / Linear Temperature Sensor
#define PIN_ENCODER_A         18  // Optical / Hall Tachometer Pulse Input

// LED Telemetry Status Indicator
#define PIN_STATUS_LED         2  // Onboard Blue LED

// =============================================================================
// PWM & TIMING CONSTANTS (LEDC PERIPHERAL)
// =============================================================================
#define PWM_FREQUENCY         15000  // 15 kHz Switching Frequency
#define PWM_RESOLUTION        10     // 10-bit resolution (0 - 1023)
#define PWM_MAX_DUTY          1023
#define DEAD_TIME_TICKS       25     // ~1.6 us Dead-time protection

#define PWM_CH_HA             0
#define PWM_CH_LA             1
#define PWM_CH_HB             2
#define PWM_CH_LB             3

#define TELEMETRY_RATE_HZ     50
const unsigned long TELEMETRY_INTERVAL_MS = 1000 / TELEMETRY_RATE_HZ; // 20 ms

// =============================================================================
// NETWORK & BACKEND CONFIGURATION
// =============================================================================
const char* WIFI_SSID     = "Your_WiFi_SSID";
const char* WIFI_PASSWORD = "Your_WiFi_Password";
const char* WS_SERVER_IP  = "192.168.1.100";  // IP address of machine running server.py
const int   WS_SERVER_PORT = 8000;
const char* WS_SERVER_PATH = "/ws/telemetry";

// =============================================================================
// GLOBAL OBJECTS & STATE VARIABLES
// =============================================================================
Adafruit_INA219 ina219;
WebSocketsClient webSocket;

bool ina219_available = false;
volatile unsigned long pulse_count = 0;
unsigned long last_rpm_calc_time = 0;
float current_rpm = 0.0f;

// Operational Command Registers
enum DriveState {
  STATE_STANDBY,
  STATE_MOTORING,
  STATE_COASTING,
  STATE_REGEN_BRAKING
};

volatile DriveState drive_state = STATE_STANDBY;
volatile float target_throttle_pct = 0.0f;
volatile float target_brake_duty_pct = 0.0f;

// Telemetry Averaging Buffers
float v_bat_filtered = 12.0f;
float i_shunt_filtered = 0.0f;
float temp_c_filtered = 28.0f;
unsigned long last_telemetry_time = 0;

// =============================================================================
// INTERRUPT SERVICE ROUTINES
// =============================================================================
void IRAM_ATTR onEncoderPulse() {
  pulse_count++;
}

// =============================================================================
// GATE DRIVER LOGIC WITH DEAD-TIME INSERTION
// =============================================================================
/*
  Note on Gate Logic with BC547 NPN Inverting Level Shifters:
  - BC547 base driven HIGH -> Collector pulled to GND -> IRF9540N Gate pulled to GND (P-MOS conducts: ACTIVE ON)
  - BC547 base driven LOW  -> Collector pulled to 12V via pull-up -> IRF9540N Gate = 12V (P-MOS OFF)
  - For Low-Side IRFZ44N (N-MOS): Direct or non-inverting driver polarity is assumed.
*/
void applyHBridgeOutputs(DriveState state, float duty_pct) {
  uint32_t duty_ticks = (uint32_t)((constrain(duty_pct, 0.0f, 100.0f) / 100.0f) * PWM_MAX_DUTY);

  switch (state) {
    case STATE_MOTORING:
      // Forward Motoring: Phase A High-Side ON, Phase B Low-Side pulsed with PWM
      ledcWrite(PWM_CH_LB, 0);                 // Low-side B OFF
      delayMicroseconds(2);                    // Hardware Dead-time
      ledcWrite(PWM_CH_HA, PWM_MAX_DUTY);      // High-side A saturated ON
      ledcWrite(PWM_CH_HB, 0);                 // High-side B OFF
      ledcWrite(PWM_CH_LA, 0);                 // Low-side A OFF
      ledcWrite(PWM_CH_LB, duty_ticks);        // Modulate Phase B low-side
      break;

    case STATE_REGEN_BRAKING:
      // Inductive Boost Regeneration:
      // Both High-Side MOSFETs driven completely OFF.
      ledcWrite(PWM_CH_HA, 0);
      ledcWrite(PWM_CH_HB, 0);
      delayMicroseconds(2);                    // Dead-time clearance

      // Pulse both Low-Side switches in synchronous boost mode
      // ON phase: Inductor charges to ground.
      // OFF phase: Inductive kickback forces current up body diodes into C1 (1000uF) & 12V rail.
      ledcWrite(PWM_CH_LA, duty_ticks);
      ledcWrite(PWM_CH_LB, duty_ticks);
      break;

    case STATE_COASTING:
    case STATE_STANDBY:
    default:
      // Safe All-Off Float State
      ledcWrite(PWM_CH_HA, 0);
      ledcWrite(PWM_CH_HB, 0);
      ledcWrite(PWM_CH_LA, 0);
      ledcWrite(PWM_CH_LB, 0);
      break;
  }
}

// =============================================================================
// SENSOR TELEMETRY ACQUISITION
// =============================================================================
void readSensors() {
  // 1. Bus Voltage via 100k/10k Divider (Factor = (100k+10k)/10k = 11.0)
  // ADC full range: 3.3V / 4095 counts * 11.0 divider factor = 0.008864 V/count
  int raw_v = analogRead(PIN_BUS_VOLTAGE_ADC);
  float v_instant = (raw_v / 4095.0f) * 3.3f * 11.05f;
  v_bat_filtered = (v_bat_filtered * 0.85f) + (v_instant * 0.15f);

  // 2. Shunt Current via INA219 (or analog fallback)
  if (ina219_available) {
    float current_mA = ina219.getCurrent_mA();
    i_shunt_filtered = (i_shunt_filtered * 0.80f) + ((current_mA / 1000.0f) * 0.20f);
  } else {
    // Shunt calculation across 0.05 Ohm: V_shunt = I * 0.05 -> I = V_shunt / 0.05
    int raw_i = analogRead(PIN_SHUNT_CURRENT_ADC);
    float v_shunt = ((raw_i - 2048) / 2048.0f) * 1.65f; // Mid-rail bi-directional offset
    float i_instant = v_shunt / 0.05f;
    i_shunt_filtered = (i_shunt_filtered * 0.85f) + (i_instant * 0.15f);
  }

  // 3. Temperature via 10k NTC Divider
  int raw_temp = analogRead(PIN_NTC_TEMP_ADC);
  float v_ntc = (raw_temp / 4095.0f) * 3.3f;
  float r_ntc = (v_ntc > 0.05f) ? (10000.0f * (3.3f - v_ntc) / v_ntc) : 10000.0f;
  // Beta approximation (B = 3950, T0 = 298.15K)
  float steinhart = r_ntc / 10000.0f;
  steinhart = log(steinhart);
  steinhart /= 3950.0f;
  steinhart += 1.0f / (25.0f + 273.15f);
  float temp_c_instant = (1.0f / steinhart) - 273.15f;
  temp_c_filtered = (temp_c_filtered * 0.90f) + (temp_c_instant * 0.10f);

  // 4. Optical RPM Calculation
  unsigned long now = millis();
  unsigned long dt_rpm = now - last_rpm_calc_time;
  if (dt_rpm >= 100) { // Calculate every 100 ms
    noInterrupts();
    unsigned long pulses = pulse_count;
    pulse_count = 0;
    interrupts();

    // 20 pulses per revolution slotted flywheel
    float revs = (float)pulses / 20.0f;
    current_rpm = (revs / (dt_rpm / 60000.0f));
    last_rpm_calc_time = now;
  }
}

// =============================================================================
// WEBSOCKET EVENT HANDLER
// =============================================================================
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] Disconnected from Telemetry Server.");
      digitalWrite(PIN_STATUS_LED, LOW);
      break;

    case WStype_CONNECTED:
      Serial.printf("[WS] Connected to Server URL: %s\n", payload);
      digitalWrite(PIN_STATUS_LED, HIGH);
      break;

    case WStype_TEXT: {
      StaticJsonDocument<256> doc;
      DeserializationError err = deserializeJson(doc, payload, length);
      if (err) return;

      const char* msg_type = doc["type"];
      if (msg_type && strcmp(msg_type, "control") == 0) {
        float throttle = doc["throttle"] | 0.0f;
        float brake = doc["brake_duty"] | 0.0f;

        target_throttle_pct = throttle;
        target_brake_duty_pct = brake;

        // Arbitration Logic: Braking takes priority over Motoring
        if (target_brake_duty_pct > 5.0f) {
          drive_state = STATE_REGEN_BRAKING;
          applyHBridgeOutputs(STATE_REGEN_BRAKING, target_brake_duty_pct);
        } else if (target_throttle_pct > 1.0f) {
          drive_state = STATE_MOTORING;
          applyHBridgeOutputs(STATE_MOTORING, target_throttle_pct);
        } else {
          drive_state = STATE_COASTING;
          applyHBridgeOutputs(STATE_COASTING, 0.0f);
        }
      }
      break;
    }

    default:
      break;
  }
}

// =============================================================================
// SETUP & INITIALIZATION
// =============================================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(PIN_STATUS_LED, OUTPUT);
  pinMode(PIN_ENCODER_A, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_ENCODER_A), onEncoderPulse, RISING);

  analogReadResolution(12); // 0 - 4095 counts
  analogSetAttenuation(ADC_11db); // 0 - 3.3V range

  // Configure 15 kHz LEDC PWM Channels
  ledcSetup(PWM_CH_HA, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcSetup(PWM_CH_LA, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcSetup(PWM_CH_HB, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcSetup(PWM_CH_LB, PWM_FREQUENCY, PWM_RESOLUTION);

  ledcAttachPin(PIN_PWM_HIGH_SIDE_A, PWM_CH_HA);
  ledcAttachPin(PIN_PWM_LOW_SIDE_A,  PWM_CH_LA);
  ledcAttachPin(PIN_PWM_HIGH_SIDE_B, PWM_CH_HB);
  ledcAttachPin(PIN_PWM_LOW_SIDE_B,  PWM_CH_LB);

  // Initialize H-Bridge to Safe Float State
  applyHBridgeOutputs(STATE_STANDBY, 0.0f);

  // Initialize I2C INA219 Sensor
  Wire.begin(21, 22);
  if (ina219.begin()) {
    ina219.setCalibration_16V_400mA();
    ina219_available = true;
    Serial.println("[✓] INA219 High-Speed Sensor Initialized.");
  } else {
    Serial.println("[!] INA219 not detected. Using calibrated analog shunt fallback.");
  }

  // Connect to WiFi
  Serial.printf("[*] Connecting to WiFi: %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int wifi_attempts = 0;
  while (WiFi.status() != WL_CONNECTED && wifi_attempts < 20) {
    delay(250);
    digitalWrite(PIN_STATUS_LED, !digitalRead(PIN_STATUS_LED));
    Serial.print(".");
    wifi_attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[✓] WiFi Connected. IP: %s\n", WiFi.localIP().toString().c_str());
    digitalWrite(PIN_STATUS_LED, HIGH);

    // Initialize WebSocket Client
    webSocket.begin(WS_SERVER_IP, WS_SERVER_PORT, WS_SERVER_PATH);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(2000);
  } else {
    Serial.println("\n[!] WiFi connection failed. Running in Serial Telemetry mode.");
  }

  last_rpm_calc_time = millis();
  last_telemetry_time = millis();
}

// =============================================================================
// MAIN 50 HZ LOOP
// =============================================================================
void loop() {
  webSocket.loop();
  readSensors();

  unsigned long now = millis();
  if (now - last_telemetry_time >= TELEMETRY_INTERVAL_MS) {
    last_telemetry_time = now;

    // Build Outbound Telemetry JSON Packet
    StaticJsonDocument<256> doc;
    doc["type"]    = "hardware_telemetry";
    doc["rpm"]     = round(current_rpm * 10.0f) / 10.0f;
    doc["v_bat"]   = round(v_bat_filtered * 100.0f) / 100.0f;
    doc["i_shunt"] = round(i_shunt_filtered * 100.0f) / 100.0f;
    doc["temp"]    = round(temp_c_filtered * 10.0f) / 10.0f;
    doc["duty"]    = (drive_state == STATE_REGEN_BRAKING) ? target_brake_duty_pct : target_throttle_pct;

    String json_string;
    serializeJson(doc, json_string);

    // Broadcast over WebSocket if connected
    if (webSocket.isConnected()) {
      webSocket.sendTXT(json_string);
    }

    // Always mirror to Hardware USB Serial for local debugging
    Serial.println(json_string);
  }
}