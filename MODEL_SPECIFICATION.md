# Trail Running Simulator  
## Model Specification v1.0

### Status
Draft specification for the first working version of the Trail Running Simulator.

---

## 1. Purpose

The Trail Running Simulator estimates trail race progression by learning how a specific runner responds to terrain, fatigue, and race context from historical `.FIT` trajectories, then applying the learned dynamics to a future race profile built from a `.GPX` file.

The simulator is designed as a **nonlinear dynamical system identification problem**:

- historical FIT files provide the training trajectories,
- the future GPX provides the race forcing profile,
- the simulator learns how the runner evolves through time and terrain,
- the simulator then integrates the learned dynamics segment by segment.

The goal is to predict:
- segment duration,
- cumulative time,
- aid-station ETA,
- finish time,
- and later, potentially, related physiological trajectories.

This first version is designed to be **testable, back-testable, and extensible**.

---

## 2. Core Modeling Principle

The simulator is not a static regression over independent rows.

It is a **trajectory-based dynamical system**.

Each historical FIT file is treated as a realized trajectory of one runner through a sequence of physiological and terrain states.

The model learns the governing evolution laws of the system from those trajectories.

For the future race, the GPX is normalized to fixed-distance segments of **50 m**. The simulator then performs a step-by-step numerical integration over that profile.

This separation is essential:

- **learning phase**: native FIT granularity, as detailed as the recorded file allows,
- **simulation phase**: normalized 50 m segments.

---

## 3. Overall Architecture

The project is organized into four logical layers.

### 3.1 Raw ingestion layer
This layer reads the source files and converts them to structured tables.

- `parser.py` converts raw `.FIT` files into message tables.
- `gpx_parser.py` converts raw `.GPX` files into a raw table of track points.

These modules do not perform system identification and do not simulate anything.

### 3.2 Feature and profile construction layer
This layer converts raw data into standardized representations.

- `features.py` converts historical FIT record data into a standardized runner profile.
- `gpx_segments.py` converts the raw GPX track into a normalized race profile and inserts aid stations as break points.

This layer prepares the data for learning and simulation.

### 3.3 Learning layer
This layer identifies the governing evolution laws of the runner.

It uses multiple historical FIT trajectories to learn how:
- cardiovascular burden evolves,
- mechanical burden evolves,
- neuromuscular burden evolves,
- interaction with terrain evolves,
- segment-level response changes with fatigue and race progression.

This is the system identification layer.

### 3.4 Simulation layer
This layer applies the learned laws to a future race profile.

Given:
- normalized 50 m race segments,
- aid stations,
- weather context,
- terrain technicality,
- strategy aggressiveness,

the simulator iterates segment by segment and estimates:
- segment duration,
- current state updates,
- cumulative ETA,
- finish time.

---

## 4. Learning and Simulation Workflow

### 4.1 Learning phase
The user uploads multiple historical `.FIT` files.

These files are used as **native-resolution trajectories**.

The learning phase must preserve the raw temporal detail of the activity because the simulator must infer:
- HR accumulation and decay,
- continuous time spent in HR zones,
- long-term and short-term fatigue mechanisms,
- trajectory-dependent response to terrain and effort.

Historical FIT data must not be collapsed prematurely to 50 m segments.

### 4.2 Simulation phase
The user uploads one future race `.GPX` file.

The GPX is normalized into fixed 50 m segments.

The user also provides aid stations and race context.

The simulator then:
- starts from the initial state,
- processes each 50 m segment in order,
- updates the state after each segment,
- accumulates total time,
- reports ETA and finish prediction.

---

## 5. Modeling Assumptions

Version 1 is based on the following assumptions:

1. A trail race can be modeled as a nonlinear dynamical system.
2. Historical FIT files contain enough information to identify the system evolution laws, provided the trajectories are sufficiently rich and numerous.
3. The future race can be represented by a 50 m normalized course profile.
4. Some variables are optional and may be absent from a FIT file.
5. Missing variables do not invalidate the simulator; they only reduce the information available for learning.
6. The model should avoid duplicating the same variable across too many states. Each variable has one canonical home.
7. The model should learn transition behavior from the input trajectories rather than relying on hand-coded recovery or fatigue rules.

---

## 6. Key Design Rule

A variable must have **one canonical home** in the model.

It may influence other states through the transition laws, but it should not be inserted as a direct input into multiple states just because it is available.

This prevents overweighting the same signal by construction.

Examples:

- Rain belongs to Race Context.
- HR zone history belongs to HR State.
- Cumulative ascent belongs to Terrain State and contributes to fatigue evolution.
- Speed belongs to Runner–Terrain Interaction.
- Segment duration is an output of the state evolution, not a state itself.

---

## 7. Version 1 Scope

Version 1 includes:
- multiple FIT file ingestion,
- standardized runner profile construction,
- GPX race profile normalization,
- aid station insertion,
- state-based learning from trajectories,
- segment-by-segment simulation over a 50 m grid.

Version 1 does not yet include:
- metabolic debt as a separate latent state,
- dynamic nutrition modeling,
- dynamic weather changes during the race,
- explicit aid-station pause modeling unless later added,
- publication-level calibration and validation.

---

## 8. Next Sections

The next part of this specification defines the state vector and all state blocks in detail:
- Terrain State
- Runner State
- Runner–Terrain Interaction
- HR State
- Fatigue State
- Race Progression
- Race Context

---

## 9. State Vector

At segment \(k\), the simulator state is represented by:

\[
x_k = (T_k, R_k, I_k, H_k, F_k, P_k, C_k)
\]

where:

- \(T_k\) = Terrain State
- \(R_k\) = Runner State
- \(I_k\) = Runner–Terrain Interaction
- \(H_k\) = HR State
- \(F_k\) = Fatigue State
- \(P_k\) = Race Progression
- \(C_k\) = Race Context

This vector describes the system at the beginning of segment \(k\).

The model learns how this state evolves through time and terrain.

---

## 10. Terrain State

### Definition

Terrain State is the deterministic geometric load of the course at segment \(k\).

It answers:

> What does the course demand right now?

Terrain State is entirely determined by the race profile.  
It does not depend on the runner.

### Variables

At segment \(k\), Terrain State is defined as:

\[
T_k = (a_k, \Delta a_k, g_k, u_k, d_k, A^+_k, A^-_k)
\]

where:

- \(a_k\) = `altitude_m`
- \(\Delta a_k\) = `altitude_delta_m`
- \(g_k\) = `grade_pct`
- \(u_k\) = `ascent_delta_m`
- \(d_k\) = `descent_delta_m`
- \(A^+_k\) = `ascent_cumul_from_start_m`
- \(A^-_k\) = `descent_cumul_from_start_m`

### Interpretation

- `altitude_m`: current elevation of the course
- `altitude_delta_m`: elevation change since the previous segment
- `grade_pct`: local slope of the segment
- `ascent_delta_m`: positive climb on the current segment
- `descent_delta_m`: positive descent on the current segment
- `ascent_cumul_from_start_m`: total climb accumulated since race start
- `descent_cumul_from_start_m`: total descent accumulated since race start

### Notes

Terrain State is fully known once the race profile is built.

It is not learned. It is read from the race profile.

The race profile is normalized to 50 m segments for simulation.

---

## 11. Runner State

### Definition

Runner State is the instantaneous physiological state of the athlete at segment \(k\).

It answers:

> What is the runner capable of producing right now?

Runner State is not the course geometry.  
It is the current internal response of the athlete.

### Variables

At minimum, Runner State includes:

- `heart_rate_bpm`
- `power`

### Optional variables

The following variables are useful when available, but they are optional:

- `cadence_spm`
- `step_length_m`
- `vertical_oscillation_mm`
- `stance_time_s`

### Notes

No variable in Runner State should be considered mandatory if the FIT file does not contain it.

The model must still run with partial data.

Runner State is dynamic and depends on:
- the previous runner state,
- terrain,
- interaction,
- fatigue,
- race progression,
- race context.

---

## 12. Runner–Terrain Interaction

### Definition

Runner–Terrain Interaction is the observable way the runner responds to the terrain at segment \(k\).

It answers:

> How did this runner actually solve this terrain, at this moment, under this fatigue state?

This is where mechanical and kinematic response lives.

### Variables

The interaction block includes:

- `speed_m_s`
- `cadence_spm`
- `step_length_m`
- `vertical_oscillation_mm`
- `stance_time_s`

### Notes

These variables are not pure runner state and not pure terrain state.

They are the result of the interaction between:
- runner physiology,
- terrain geometry,
- fatigue,
- and race context.

This block is also where technicity is learned indirectly.

For example:
- the same descent may produce very different speed and cadence depending on whether the terrain is smooth or technical,
- the same uphill may produce different step length and stance time depending on fatigue and slope.

### Optional variables

As with Runner State, some interaction variables may be missing in some FIT files.  
The simulator must tolerate missing interaction variables.

---

## 13. HR State

### Definition

HR State is the physiological cardiovascular state of the runner at a given segment.

It describes both:
- the instantaneous cardiovascular demand,
- and the accumulated cardiovascular stress from the start of the race.

HR State is the bridge between Runner State and Fatigue State.

### 13.1 Static HR profile

The user provides:
- `HR_rest`
- `HR_max`

From these values, the system computes a six-zone HR model, which the user can edit.

The six zones are:

- Z1: Recovery
- Z2: Fundamental endurance
- Z3: Tempo
- Z4: Threshold
- Z5: VO₂max
- Z6: Max effort / Sprint

### 13.2 Instantaneous HR state

At every segment, the model knows:
- current heart rate
- current HR zone

These are the current cardiovascular intensity descriptors.

### 13.3 HR history

For each zone \(i \in \{1,\dots,6\}\), the model stores:

- `time_in_zone_i`
- `fraction_time_in_zone_i`
- `continuous_time_spend_in_zone_i`

#### Meaning of these quantities

- `time_in_zone_i`: total cumulative time spent in zone \(i\)
- `fraction_time_in_zone_i`: cumulative time in zone \(i\) divided by elapsed race time
- `continuous_time_spend_in_zone_i`: current uninterrupted time spent in zone \(i\)

The following rule always holds:

\[
continuous\_time\_spend\_in\_zone_i \le time\_in\_zone_i
\]

### 13.4 Latent HR debt

HR debt is not directly observed.

It is inferred from:
- current HR,
- current HR zone,
- cumulative zone exposure,
- continuous zone exposure,
- and the overall race history.

HR debt accumulates when the runner spends time in high-intensity zones and decays when intensity decreases.

The exact decay law is learned from the historical FIT trajectories.

---

## 14. Fatigue State

### Definition

Fatigue State is the latent cumulative burden created by the race history, which changes how the runner responds to the same terrain and the same effort over time.

Fatigue State is not a single scalar.

It is a vector of coupled debts.

### 14.1 Fatigue debts

In Version 1, Fatigue State contains three latent debts:

- Cardiovascular Debt
- Mechanical Debt
- Neuromuscular Debt

So:

\[
F_k = (D^C_k, D^M_k, D^N_k)
\]

where:

- \(D^C_k\) = Cardiovascular Debt
- \(D^M_k\) = Mechanical Debt
- \(D^N_k\) = Neuromuscular Debt

### 14.2 Cardiovascular Debt

Cardiovascular Debt reflects the accumulated physiological strain associated with:
- HR zone exposure,
- continuous time in HR zones,
- overall intensity history,
- and HR recovery dynamics.

It is strongly linked to HR State.

### 14.3 Mechanical Debt

Mechanical Debt reflects the cost of:
- climbing,
- descending,
- braking,
- impact,
- eccentric load,
- and technical terrain interaction.

It is related to:
- cumulative ascent,
- cumulative descent,
- slope,
- and mechanical behavior on the terrain.

### 14.4 Neuromuscular Debt

Neuromuscular Debt reflects degradation in movement quality and coordination.

It is related to:
- cadence drift,
- step length reduction,
- vertical oscillation changes,
- stance time changes,
- and repeated technical terrain exposure.

### 14.5 Important note

Version 1 does **not** include a separate metabolic debt state.

Metabolic variables such as nutrition and hydration are too uncertain to model directly from FIT trajectories alone.

Temperature belongs to Race Context, not to Fatigue State.

---

## 15. Race Progression

### Definition

Race Progression describes where the runner is in the race and what remains to be completed.

It is deterministic.

It does not depend on the runner’s physiological condition.

### Variables

At segment \(k\), Race Progression includes:

- `distance_from_start_m`
- `time_from_start_s`
- `fraction_of_race_completed`
- `remaining_distance_m`
- `remaining_ascent_m`
- `remaining_descent_m`
- `distance_since_last_aid_station_m`
- `distance_to_next_aid_station_m`

### Notes

Race Progression is updated mechanically from one segment to the next.

It is not learned as a latent variable.

It is part of the state because the runner’s strategy and response depend on how much race remains.

---

## 16. Race Context

### Definition

Race Context is the set of external and subjective conditions known before the race that modify how the system evolves, without being part of the runner, the terrain geometry, or the accumulated debts themselves.

Race Context has one canonical home for each variable, so that the same input is not duplicated across multiple states.

### Variables

Race Context contains:

- `weather_rain_level`
- `weather_temperature_c`
- `terrain_technicality`
- `strategy_aggressiveness`

### 16.1 Weather rain level

A three-level categorical variable:

- `no_rain`
- `light_rain`
- `heavy_rain`

Meaning:
- `no_rain`: no specific traction penalty
- `light_rain`: mainly affects descents, with slippery footing
- `heavy_rain`: affects both ascents and descents

### 16.2 Weather temperature

A continuous variable:

- `weather_temperature_c` in \([-20^\circ C, +45^\circ C]\)

This is entered in the UI with a practical step of 5°C.

The temperature modifies:
- thermal strain,
- hydration difficulty,
- cardiovascular cost,
- and recovery behavior.

### 16.3 Terrain technicality

A subjective scale from 1 to 5:

- 1 = very runnable
- 5 = highly technical

This represents the pre-race expected technical difficulty of the course.

### 16.4 Strategy aggressiveness

A scale from 1 to 3:

- 1 = conservative
- 2 = usual
- 3 = aggressive

This represents the runner’s intended pacing strategy for the race.

### Notes

Race Context is fixed for the race in Version 1.

It influences the evolution of the system, but it is not itself dynamically updated during the simulation.

---

## 17. Summary of State Responsibilities

### Terrain State
What the course demands geometrically.

### Runner State
What the athlete is physically producing right now.

### Runner–Terrain Interaction
How the athlete actually responds mechanically and kinematically to the course.

### HR State
How the cardiovascular system behaves, accumulates stress, and recovers.

### Fatigue State
How the accumulated debts evolve and alter future response.

### Race Progression
Where the runner is in the race and how much remains.

### Race Context
The external and subjective race conditions that shape the system evolution.

---

## 18. Next Section

The next part of this specification defines the transition laws:
- what is deterministic,
- what is learned,
- and how the system evolves from one 50 m segment to the next.
