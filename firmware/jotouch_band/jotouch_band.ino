/*
 * JoTouch Band Firmware
 *
 * Reads 4 FSR channels at exactly 100 Hz and reports:
 *   fsr0,fsr1,fsr2,fsr3,led_state\r\n
 *
 * IMPORTANT: never use Arduino String — it fragments the AVR heap and causes
 * silent output corruption after ~5 s at 100 Hz.  All formatting is done with
 * a fixed-size char buffer and itoa/sprintf.
 *
 * LED pattern (host-triggered, see ADR 003):
 *   Idle:             LED OFF, FSR still streaming.  Host has not yet armed.
 *   Blink mode ('B'):  1 Hz periodic blink, 100 ms ON.  No PRBS preamble.
 *                       Used for ROI calibration and sync check — the operator
 *                       can see the LED and the camera can detect brightness
 *                       changes without wasting the PRBS preamble.
 *   Armed mode ('S'):  Phase 1 (0–6.3 s):  PRBS preamble — 63-bit m-sequence
 *                       (x^6+x+1), 100 ms per chip.  Unambiguous sync acquisition.
 *                       Phase 2 (after):    1 Hz periodic blink, 100 ms ON.
 *
 * The host sends 'B' to start blink-only mode (for ROI calibration / sync check)
 * or 'S' to arm the PRBS preamble (for recording).  Before either is received,
 * s_start_ms == 0 and get_led_state() returns false (LED OFF).
 *
 * The LED state reported in each CSV line is sampled at the same moment as
 * the FSR reads so the column is always in sync.
 */

// ── Timing ────────────────────────────────────────────────────────────────────
static const unsigned long SAMPLE_INTERVAL_MS = 10;  // 100 Hz
static unsigned long s_last_sample_ms = 0;
static unsigned long s_start_ms = 0;  // 0 = not yet armed; set on receipt of 'B' or 'S'
static const unsigned long NOT_ARMED = 0;
static bool s_skip_prbs = false;  // true = blink-only ('B'), false = PRBS+blink ('S')

// ── LED ───────────────────────────────────────────────────────────────────────
static const int LED_PIN        = 8;
static const unsigned long LED_ON_MS     = 100;   // ON duration (periodic mode)
static const unsigned long LED_PERIOD_MS = 1000;  // full cycle (periodic mode)

// PRBS preamble: 63-bit m-sequence from x^6 + x + 1 (taps [1,6])
static const int PRBS_LEN = 63;
static const unsigned long PRBS_CHIP_MS = 100;  // 100 ms per chip
static const unsigned long PRBS_DURATION_MS = 6300;  // 63 * 100
static const uint8_t PRBS_SEQ[63] = {
    1,0,1,0,1,0,1,1,0,0, 1,1,0,1,1,1,0,1,1,0,
    1,0,0,1,0,0,1,1,1,0, 0,0,1,0,1,1,1,1,0,0,
    1,0,1,0,0,0,1,1,0,0, 0,0,1,0,0,0,0,0,1,1,
    1,1,1
};

// ── Sensors ───────────────────────────────────────────────────────────────────
static const int N_SENSORS   = 4;
static const int SENSOR_PINS[N_SENSORS] = {0, 1, 2, 3};  // A0-A3

// ── Output buffer ─────────────────────────────────────────────────────────────
static char s_line[32];

// ── Helpers ───────────────────────────────────────────────────────────────────

static bool append_uint(char* buf, int buf_size, int* pos, unsigned int val) {
    char tmp[6];
    int len = 0;
    if (val == 0) {
        tmp[len++] = '0';
    } else {
        unsigned int v = val;
        while (v > 0) { tmp[len++] = '0' + (v % 10); v /= 10; }
        for (int i = 0, j = len - 1; i < j; i++, j--) {
            char c = tmp[i]; tmp[i] = tmp[j]; tmp[j] = c;
        }
    }
    if (*pos + len >= buf_size - 1) return false;
    for (int i = 0; i < len; i++) buf[(*pos)++] = tmp[i];
    return true;
}

static bool get_led_state(unsigned long now) {
    if (s_start_ms == NOT_ARMED) {
        // Host has not yet sent 'B' or 'S' — LED stays OFF, FSR still streams.
        return false;
    }
    unsigned long elapsed = now - s_start_ms;
    if (!s_skip_prbs && elapsed < PRBS_DURATION_MS) {
        // PRBS preamble (armed mode only)
        int chip = (int)(elapsed / PRBS_CHIP_MS) % PRBS_LEN;
        return PRBS_SEQ[chip] ? true : false;
    } else {
        // Periodic 1 Hz blink (both blink-only and armed mode after PRBS)
        unsigned long phase = elapsed % LED_PERIOD_MS;
        return (phase < LED_ON_MS);
    }
}

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    analogReference(EXTERNAL);
    pinMode(LED_PIN, OUTPUT);
    delay(1000);
    // Do NOT set s_start_ms here — wait for host 'B' or 'S' command (ADR 003).
    // FSR samples start flowing immediately so the host can verify data
    // is arriving during pre-collect tests.  The LED column reads 0
    // until the host arms the LED (blink-only or PRBS preamble).
    s_start_ms = NOT_ARMED;
    s_skip_prbs = false;
    s_last_sample_ms = millis();
}

// ── Loop ──────────────────────────────────────────────────────────────────────

void loop() {
    unsigned long now = millis();

    // ── Non-blocking serial command check ───────────────────────────────────
    // Serial.available() returns immediately (0 if nothing queued), so the
    // 100 Hz sample cadence is unaffected.  Commands:
    //   'B' = start blink-only mode (periodic 1 Hz, no PRBS) — for ROI / sync check
    //   'S' = arm PRBS preamble then periodic blink — for recording
    if (Serial.available() > 0) {
        int b = Serial.read();
        if (b == 'S') {
            // Arm the PRBS preamble.  Reset the sample timer so the first
            // post-arm sample aligns with t=0 of the preamble.
            s_start_ms = millis();
            s_skip_prbs = false;
            s_last_sample_ms = s_start_ms;
        } else if (b == 'B') {
            // Start blink-only mode (no PRBS preamble).  Used for ROI
            // calibration and sync check so the operator can see the LED
            // without wasting the PRBS preamble.
            s_start_ms = millis();
            s_skip_prbs = true;
            s_last_sample_ms = s_start_ms;
        }
        // Drain any extra bytes so they don't interfere with the next command.
        while (Serial.available() > 0) { Serial.read(); }
    }

    // ── LED update (runs every loop iteration for accurate timing) ────────────
    bool led_on = get_led_state(now);
    digitalWrite(LED_PIN, led_on ? HIGH : LOW);

    // ── Sample at 100 Hz ──────────────────────────────────────────────────────
    if (now - s_last_sample_ms < SAMPLE_INTERVAL_MS) return;
    s_last_sample_ms += SAMPLE_INTERVAL_MS;

    // Read LED state at the exact sample moment
    led_on = get_led_state(now);

    // Build CSV line without heap allocation
    int pos = 0;
    for (int i = 0; i < N_SENSORS; i++) {
        int val = analogRead(SENSOR_PINS[i]);
        append_uint(s_line, sizeof(s_line), &pos, (unsigned int)val);
        s_line[pos++] = ',';
    }
    s_line[pos++] = led_on ? '1' : '0';
    s_line[pos++] = '\r';
    s_line[pos++] = '\n';
    s_line[pos]   = '\0';

    Serial.write((const uint8_t*)s_line, pos);
}
