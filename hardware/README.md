# Calibration interface

The canonical file is `src/corgi_rl/specs/corgi.json`; generated MJCF is an output.
All current values are **engineering estimates**, including motor IDs and signs.
There is no chosen supplier, production BOM or verified torque budget yet.

Units: m, kg, s, rad, N·m. Body axes: x forward, y left, z up. Quaternion: wxyz.
Joint order: FL, FR, RL, RR; hip abduction, thigh pitch, knee pitch for each leg.
The hip axes are +x; thigh/knee axes are +y in their local frames.

For each real joint measure encoder sign and zero. The implemented conversion is
`q_hardware = direction * q_model + zero_rad`; the inverse is
`q_model = direction * (q_hardware - zero_rad)`. Position limits are in model
coordinates. No current function sends motor commands.

The shared JSON accepts per-joint PD gain, effort cap, armature and frictionloss.
Geometry and mass generate MuJoCo inertias from primitive shapes. Optional
per-body `inertial_overrides` replace these with measured mass, center of mass
and diagonal principal inertia in the body frame. Rotated/non-diagonal measured
inertia needs an extension before import; do not silently drop its off-diagonals.

Calibration sequence:

1. Obtain supplier motor protocol, supply voltage, continuous/peak torque,
   speed, thermal limits, feedback fields and dimensions.
2. Measure link mass, center of mass, inertia and joint axes. Fit geometry and
   regenerate the model. Check collision clearance and mechanical limits.
3. Test one suspended joint at low current: sign, zero, angle limits, command
   delay, tracking, friction, backlash and loaded torque-speed response.
4. Fit actuator parameters from logs. Extend the actuator model for effects
   not represented here; then add measured domain randomization ranges.
5. Recheck model/observation/action consistency, retrain, and validate in
   suspended or supported tests before ground locomotion.

Not yet modeled: bus latency, target slew limits, torque-speed/voltage curves,
backlash, thermal state, sensor noise, battery sag, flexible links. `unmeasured`
records these gaps. Their omission prevents any sim-to-real readiness claim.
