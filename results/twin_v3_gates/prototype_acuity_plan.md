# Feasibility test 2: does patient acuity create headroom? (prototype only)

**Written 2026-10-01, before the prototype is implemented or run.** User-requested quick test,
after reactive levers (test 1) showed no headroom.
- **Not part of twin-v3:** prototype script only; no change to `src/`, the protocol or any gate.
- **Aim:** give health, transport and power separate stakes in who dies, instead of everything
  flowing through "beds lost to facility damage".

**Mechanics (anchored in sources; remaining choices are disclosed assumptions):**
1. **Triage.** 20% of new hospital arrivals are critical. Sources: ESI handbook, expected ESI 1–2
   = 21–33%; NHAMCS 2017 observed ~11%. A critical patient's safe wait is 1 h (non-critical 4 h,
   unchanged). Hospitals admit critical patients first (standard triage).
2. **Time to care.** 30% of emergency calls are critical (assumption: emergency calls skew to
   higher acuity). A critical casualty must reach a bed within 1 h of the call (the "golden hour";
   trauma mortality rises about 1–2% per minute of response time); otherwise they die, on the road
   or in the queue. Every casualty's waiting clock starts at the call, not at hospital arrival.
   Dispatch centres triage calls, so critical calls get earlier slack (applies to every policy).
3. **Critical care needs power.** ICU patients (`icu_occupied`) in a hospital without power (its
   own flooding, or grid loss after the backup runs out) die at 0.5% per hour. This is derived
   from MIOT Chennai 2015: 18 of 75 ventilated patients over about 48 h. An urgent transfer from
   such a hospital moves an ICU patient first.

**Rainfall:** single cell (unchanged). Two-cell rainfall is tested only if acuity alone shows no
headroom.

**Conditions** (the same 10 moderate/severe train scenarios as test 1, seed k = 0, all inside the
prototype):
1. **RB-S** (acuity-blind rules).
2. **Simple acuity-aware rule:**
   - RB-S, plus health requests urgent transfers of ICU patients from hospitals without power;
   - transport sends critical casualties to the nearest functioning hospital (time to care).
3. **Oracle:** every hour, perfect foresight over 45 strategies = health {RB-S, ICU-evacuation,
   E-coordinated} × power {RB-S, D-nearest, E-coordinated} × transport {RB-S, acuity
   destination, A-fixed, C-calls-first, C-transfers-first}, scored on an exact env copy over 6 h
   (the strategy for 1 h, then rule 2).

**Reading (fixed now):**
- **Acuity value:** how deaths compare across 1 and 2.
- **Headroom:** if (3) is ≥ 15% below the better of (1) and (2) in mean deaths, acuity creates
  headroom a learner could exploit. Otherwise it does not.
