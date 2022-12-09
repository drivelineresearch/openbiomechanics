# OpenBiomechanics Project (OBP) Documentation

![IMG_5797.JPG](imgs/IMG_5797.jpg)

Homepage: [OpenBiomechanics.org](https://openbiomechanics.org)

# C3D

Cleaned C3D files are provided at `~\baseball_pitching\data\c3d` for those who wish to conduct their own analysis process from start to finish. C3Ds are separated into athlete-specific folders with static model files also provided.

## File Naming

All C3Ds follow a common naming convention. 

`USERid_SESSIONid_HEIGHT_WEIGHT_PITCHNUMBER_PITCHTYPE_PITCHSPEED.c3d`

`USERid` = unique athlete identifier

`SESSIONid` = unique session identifier

`HEIGHT` = body height in inches (body height in meters is also provided in the metadata CSV)

`WEIGHT` = bodyweight in pounds (body mass in kilograms is also provided in the metadata CSV)

`PITCHNUMBER` = pitch number from the athlete’s assessment

`PITCHTYPE` = type of pitch thrown

- FF = fastball

`PITCHSPEED` = speed of the pitch thrown in miles per hour to one decimal place (with the decimal place removed)

- ex. `~_905.c3d` would be a pitch speed of 90.5 mph; `~_950.c3d` would be a pitch speed of 95.0 mph

## Marker Set

**Front View**

![imgs/image2.png](imgs/image2.png)

*NOTE: LIC and RIC (Iliac Crest markers) are no longer used and may not be present in some C3D files. Additionally, RKNEE2 and LKNEE2 are labeled as RMKNE and LMKNE respectively.*

**Back View**

![imgs/image3.png](imgs/image3.png)

*NOTE: LIC and RIC (Iliac Crest markers) are no longer used and may not be present in some C3D files. Additionally, RKNEE2 and LKNEE2 are labeled as RMKNE and LMKNE respectively.*

## Force Plates

Ground reaction force data were collected using three force plates embedded underneath the pitching mound turf surface (average turf thickness ~ 0.5 inches).

![force_plates.jpeg](imgs/force_plates.jpeg)

## Global Coordinate System

The global (laboratory) reference frame is such that (+) x points from second base towards home plate, (+) points towards first base, and (+) z points upward

![gcs.png](imgs/gcs.png)

# Full Signal

We first calculate joint angles, velocities, forces, and moments according to the right hand rule. We then perform select adjustments to better match coach/player intuition (righty/lefty symmetry, sign negations, adding or subtracting 90 degrees, etc.). Forces and Moments are internal forces and moments.

Below is a list of joint mechanics, their components, and conventions. Where joint kinematics are constrained by the model, corresponding kinetic conventions are provided. 

Unadjusted time series data are also available upon request.

## Processed Joint Kinematic/Kinetic Conventions

Processed full signal data were filtered with a 4th order Butterworth low pass filter with a cutoff frequency of 20.0 Hz.

### Kinematics (Joint Angles, Joint Velocities)

| Joint | C1 (”_X”) | C2 (”_Y”) | C3 (”_Z”) |
| --- | --- | --- | --- |
| Wrist | Extension (+)/Flexion (-) | Ulnar (+)/Radial (-) Deviation | Constrained |
| Elbow | Flexion (+)/Extension (-) | Constrained | Pronation (+)/Supination (-) |
| Shoulder | Horizontal Ab (+)/Adduction (-) | Add (+)/Abduction (-) | External (+)/Internal (-) Rotation |
| Pelvis | Anterior (+)/Posterior (-) Tilt | Throwing Side (+)/Glove Side (-) Hip Hike (Lateral Flexion) | Axial Rotation Towards (+)/Away (-) from home plate |
| Torso | Flexion (+)/Extension (-) | Glove Side (+)/Throwing Side (-) Lateral Flexion | Axial Rotation Towards (+)/Away (-) from home plate |
| Torso-Pelvis | Extension (+)/Flexion (-) | Glove Side (+)/Throwing Side (-) Lateral Flexion | Hip-Shoulder Separation (+)/Closing (-) |
| Hip | Extension (+)/Flexion (-) | Ab (+)/Adduction (-) | External (+)/Internal (-) Rotation |
| Knee | Flexion (+)/Extension (-) | Constrained | Constrained |
| Ankle | Dorsi (+)/Plantar (-) Flexion | Ab (+)/Adduction (-) | Inversion (+)/Eversion (-) |

### Forces

| Joint | C1 (”_X”) | C2 (”_Y”) | C3 (”_Z”) |
| --- | --- | --- | --- |
| Wrist | Lateral (+)/ Medial(-) | Anterior (+)/Posterior (-) | Compression (+)/Distraction (-) |
| Elbow | Lateral (+)/ Medial (-) | Anterior (+)/ Posterior (-) | Compression (+)/Distraction (-) |
| Shoulder | Rel. upper arm: ? Rel. thorax: Lateral (+)/Medial (-) | Rel. upper arm: ? Rel. thorax: Anterior (+)/Posterior (-) | Rel upper arm: Compression (+)/Distraction (-). Rel thorax: Superior (+)/Inferior (-) |
| Pelvis | NA | NA | NA |
| Torso | NA | NA | NA |
| Torso-Pelvis | NA | NA | NA |
| Hip | Rel thigh: ? Rel pelvis: Lateral (+)/Medial (-) | Rel thigh: ? Rel pelvis: Anterior (+)/Posterior (-) | Rel thigh: Compression (+)/Distraction (-). Rel pelvis: Superior (+)/Inferior (-) |
| Knee | Lateral (+)/Medial (-) | Anterior (+)/Posterior (-) | Compression (+)/Distraction (-) |
| Ankle | Lateral (+)/Medial (-) | Anterior (+)/Posterior (-) | Superior (+)/Inferior (-) |

### Moments

| Joint | C1 (”_X”) | C2 (”_Y”) | C3 (”_Z”) |
| --- | --- | --- | --- |
| Wrist | Flexion (+)/Extension (-) | Ulnar (+)/Radial (-) Deviation | NA |
| Elbow | Flexion (+)/Extension (-) | Varus (+)/Valgus (-) | Pronation (+)/Supination (-) |
| Shoulder | Rel upper arm: ? Rel thorax:  ? | Rel upper arm: ? Rel. thorax: Ab (+)/ Adduction (-) | Rel upper arm: Internal (+)/External (-) Rotation. Rel thorax: Horizontal Add (+)/Abduction (-) |
| Pelvis | NA | NA | NA |
| Torso | NA | NA | NA |
| Torso-Pelvis | NA | NA | NA |
| Hip | Rel thigh: ? Rel pelvis: Extension (+)/Flexion (-) | Rel thigh: ? Rel pelvis: ? | Rel thigh: Internal (+)/External (-) Rotation. Rel pelvis: ? |
| Knee | Flexion (+)/Extension (-) | Varus (+)/Valgus (-) | External (+)/Internal (-) Rotation |
| Ankle | Plantar (+)/Dorsi (-) Flexion | Eversion (+)/Inversion (-) | Ab(+)/Adduction (-) |

## Processed Ground Reaction Force Conventions

Ground reaction force data were filtered with a 4th order Butterworth low pass filter with a cutoff frequency of 40.0 Hz

| Force Plate | X | Y | Z |
| --- | --- | --- | --- |
| FP1/3 | + = Braking (Posterior) | + = Lateral | + = Superior |
| FP2 | + = Push off (Anterior) | + = Lateral | + = Superior |

## Table Schema

Full signal data are broken up into six large CSV files:

- `energy_flow`: energy transfer, energy generation, energy absorption
- `force_plate`: rear leg, lead leg ground reaction forces
- `forces_moments`: joint forces, joint moments
- `joint angles`: joint angles
- `joint_velos`: joint angular velocities
- `landmarks`: joint center positions

Tables may be joined using `session_pitch` + `time`. Please note that force plate and marker-derived data are provided at their own respective measurement rates (360 Hz and 1,080 Hz, respectively). Therefore, we recommend caution when joining force plate and marker-derived data to avoid potential data loss.

The times at which common events occurred are also joined in each table for convenience. Provided events are foot contact (10% bodyweight), foot plant (100% bodyweight), maximum external rotation (layback), ball release, and maximum internal rotation.

By default, forces and moments are expressed in the proximal segment’s coordinate system (ex. `lead_ankle_moment_x` represents the moment at the lead ankle about the shank’s x-axis). Exceptions are made for the forces and moments at the shoulders and hips. For these joints, two sets of forces and two sets of moments are provided. The first set is the joint force/moment resolved in the proximal segment’s coordinate system (ex. shoulder force along the thorax’s x, y, and z axes). The second set is the joint force/moment resolved in the distal segment’s coordinate system (ex. shoulder force along the upper arm’s x, y, and z axes). For these “double dipped” joint kinetics, we follow the naming convention `joint_referenceSegment_kineticType_element`. For example, `shoulder_upper_arm_moment_z` represents the moment at the shoulder about the upper arm’s longitudinal (z) axis and `shoulder_thorax_moment_z` represents the moment at the shoulder resolved about the thorax superior-inferior (z) axis.

One potential research project is to process the provided C3D using your own pipeline and compare your results with ours. Self-processed data from C3D files may be linked to the provided full signal data through the metadata CSV located at `~\baseball_pitching\data\metadata.csv`. 

# Point of Interest

Kinematic, kinetic, and energetic metrics commonly referenced in biomechanical analyses are arranged into a point-of-interest (POI) CSV. We attempted to name variables so their definitions would be clear, however, this may not always be the case.

```yaml
"session_pitch": unique pitch ID
"session": unique session ID
"p_throws": pitcher handedness
"pitch_type": pitch type
"pitch_speed_mph": pitch speed (miles per hour)
"max_shoulder_internal_rotational_velo": peak shoulder internal rotation velocity (deg/s)
"max_elbow_extension_velo": peak elbow extension velocity (deg/s)
"max_torso_rotational_velo": peak torso axial rotation (Z) velocity (deg/s)
"max_rotation_hip_shoulder_separation": peak separation between thorax and pelvis (deg)
"max_elbow_flexion": peak elbow flexion angle (deg)
"max_shoulder_external_rotation": peak shoulder external rotation (layback) (deg)
"elbow_flexion_fp": elbow flexion angle at foot plant (deg)
"elbow_pronation_fp": wrist pronation angle at foot plant (deg)
"rotation_hip_shoulder_separation_fp": hip-shoulder separation agnle at foot plant (deg)
"shoulder_horizontal_abduction_fp": shoulder horizontal abduction angle at foot plant (deg)
"shoulder_abduction_fp": shoulder abduction angle at foot plant (deg)
"shoulder_external_rotation_fp": shoulder external rotation angle at foot plant (deg)
"lead_knee_extension_angular_velo_fp": lead knee extension angular velocity at foot plant (deg/s)
"lead_knee_extension_angular_velo_br": lead knee extension angular velocity at ball release (deg/s)
"lead_knee_extension_angular_velo_max": peak knee extension angular velocity (deg/s)
"torso_anterior_tilt_fp": trunk flexion at foot plant (deg)
"torso_lateral_tilt_fp": trunk lateral flexion at foot plant (deg)
"torso_rotation_fp": trunk axial rotation at foot plant (deg)
"pelvis_anterior_tilt_fp": pelvis anterior tilt at foot plant (deg)
"pelvis_lateral_tilt_fp": pelvis lateral tilt at foot plant (deg)
"pelvis_rotation_fp": pelvis axial rotation at foot plant (deg)
"max_cog_velo_x": peak center of gravity velocity towards home plate (m/s)
"torso_rotation_min": peak torso counterrotation angle (deg)
"max_pelvis_rotational_velo": peak pelvis axial rotation angular velocity (deg/s)
"glove_shoulder_horizontal_abduction_fp": glove shoulder horizontal abduction angle at foot plant (deg)
"glove_shoulder_abduction_fp": glove shoulder abduction angle at foot plant (deg)
"glove_shoulder_external_rotation_fp": glove shoulder external rotation angle at foot plant (deg)
"glove_shoulder_abduction_mer": glove shoulder abduction angle at maximum external rotation (deg)
"elbow_flexion_mer": elbow flexion angle at maximum external rotation (deg)
"torso_anterior_tilt_mer": trunk flexion at maximum external rotation (deg)
"torso_lateral_tilt_mer": trunk lateral flexion at maximum external rotation (deg)
"torso_rotation_mer": trunk axial rotation at maximum external rotation (deg)
"elbow_varus_moment": peak elbow varus moment (Nm)
"shoulder_internal_rotation_moment": peak shoulder internal rotation moment (Nm)
"torso_anterior_tilt_br": trunk flexion angle at ball release (deg)
"torso_lateral_tilt_br": trunk lateral flexion angle at ball release (deg)
"torso_rotation_br": trunk axial rotation angle at ball release (deg)
"lead_knee_extension_from_fp_to_br": difference in lead knee extension angle between foot plant and ball release (deg)
"cog_velo_pkh": center of gravity velocity towards home plate at peak knee height (m/s)
"stride_length": stride length (%body height)
"stride_angle": stride angle (+ = cross-body stride) (deg)
"arm_slot": arm slot (global forearm projection angle) (deg)
"timing_peak_torso_to_peak_pelvis_rot_velo": time from peak pelvis angular velocity to peak torso angular velocity (s)
"max_shoulder_horizontal_abduction": peak shoulder horizontal abduction angle (deg)
"shoulder_transfer_fp_br": energy transfer across throwing shoulder between foot plant and ball release (J)
"shoulder_generation_fp_br": energy generation across throwing shoulder between foot plant and ball release (J)
"shoulder_absorption_fp_br": energy absorption across throwing shoulder between foot plant and ball release (J)
"elbow_transfer_fp_br": energy transfer across throwing elbow between foot plant and ball release (J)
"elbow_generation_fp_br": energy generation across throwing elbow between foot plant and ball release (J)
"elbow_absorption_fp_br": energy absorption across throwing elbow between foot plant and ball release (J)
"lead_hip_transfer_fp_br": energy transfer across lead hip between foot plant and ball release (J)
"lead_hip_generation_fp_br": energy generation across lead hip between foot plant and ball release (J)
"lead_hip_absorption_fp_br": energy absorption across lead hip between foot plant and ball release (J)
"lead_knee_transfer_fp_br": energy transfer across lead knee between foot plant and ball release (J)
"lead_knee_generation_fp_br": energy generation across lead knee between foot plant and ball release (J)
"lead_knee_absorption_fp_br": energy absorption across lead knee between foot plant and ball release (J)
"rear_hip_transfer_pkh_fp": energy transfer across rear hip between foot plant and ball release (J)
"rear_hip_generation_pkh_fp": energy generation  across rear hip between foot plant and ball release (J)
"rear_hip_absorption_pkh_fp": energy absorption across rear hip between foot plant and ball release (J)
"rear_knee_transfer_pkh_fp": energy transfer across rear knee between foot plant and ball release (J)
"rear_knee_generation_pkh_fp": energy generation across rear knee between foot plant and ball release (J)
"rear_knee_absorption_pkh_fp": energy absorption across rear knee between foot plant and ball release (J)
"pelvis_lumbar_transfer_fp_br": energy transfer out of pelvis towards trunk between foot plant and ball release (J)
"thorax_distal_transfer_fp_br": energy transfer out of trunk towards throwing shoulder between foot plant and ball release (J)
"rear_grf_x_max": peak anterior push-off ground reaction force (N)
"rear_grf_y_max": peak lateral push-off ground reaction force (N)
"rear_grf_z_max": peak vertical ground reaction force on the rear leg (N)
"rear_grf_mag_max": peak resultant ground reaction force on the rear leg (N)
"rear_grf_angle_at_max": rear ground reaction force vector projection angle at the time of peak ground reaction force magnitude (deg)
"lead_grf_x_max": peak braking ground reaction force (N)
"lead_grf_y_max": peak lateral braking ground reaction force (N)
"lead_grf_z_max": peak vertical ground reaction force on the lead leg (N)
"lead_grf_mag_max": np.nanmax(lead_grf_mag),
"lead_grf_angle_at_max": lead ground reaction force vector projection angle at the time of peak ground reaction force magnitude (deg)
"peak_rfd_rear": peak rate of force development on the rear leg (N/s)
"peak_rfd_lead": peak rate of force development on the lead leg (N/s)
```
