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
