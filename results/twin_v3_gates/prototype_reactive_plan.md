# Feasibility test: do reactive, capacity-restoring levers create headroom? (prototype only)

**Written 2026-10-01, before the prototype is implemented or run.** User-requested quick test.
- **It is not part of twin-v3:** no change to `src/`, the frozen action space, the protocol or any
  gate. Everything lives in `scripts/prototype_reactive_levers.py`.
- **It is not forecasting:** both resources are dispatched only in response to observed failure or
  flooding.

**Resources (scarce, reactive):**
- **1 mobile generator:** sent to a hospital. While on site, the hospital's power-edge buffer is
  kept full (on-site power), so grid loss no longer reduces its functional level. It cannot fix a
  hospital that is itself flooded or damaged.
- **1 dewatering pump:** sent to a substation or water pump. While on site, flood damage to that
  facility is suppressed, so a repair holds while the site is still flooded.

Both start at the utility depot (S0's road node), travel on the current flooded road network
(travel time fixed at dispatch, like the crew) and can be reassigned.

**Conditions** (10 moderate/severe train scenarios, evaluation seed k = 0; other sectors =
frozen RB-S):
1. **RB-S:** no resources used (the existing gate runs).
2. **RB-S + simple reactive rule:**
   - generator to the hospital whose power buffer is being drawn with the least time left; kept
     while that hospital's grid supply is still failing;
   - pump to the flooded, damaged substation or pump with the most dependants; kept while that
     site is still flooded.
3. **Oracle:** every hour, 4 generator choices {keep, H1, H2, H3} × 7 pump choices {keep, the 3
   substations, the 3 pumps} = 28, scored with perfect foresight on an exact env copy over 6 h
   (the choice, then the rule). The best is executed.

**Reading (fixed now):**
- **Lever value:** (2) vs (1), mean deaths.
- **Headroom:** (3) vs (2). If (3) is ≥ 15% lower in mean deaths than (2), reactive levers create
  headroom a learner could exploit. Otherwise they do not, in this form.
