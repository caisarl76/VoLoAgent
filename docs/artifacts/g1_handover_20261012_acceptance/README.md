# Handover acceptance, 12–13 October 2026

The checkpoint loads and predicts plausible closed fingers at the sampled grasp
frames and open fingers at the sampled offer frames. The native SONIC ready
return reached its target in simulation. Desk clearance failed the geometric
screen, and the repeating physical handover remains unqualified.

Watch the [ready-return video](ready-return.mp4), or inspect its
[four-frame preview](ready-return-preview.jpg). It shows actual saved MuJoCo
states from the native controller, with the robot standing on its feet and
virtual support disabled. The bottle stays on the desk. This is a ready-return
component test from default standing; it contains no VLA pickup, offering or
human removal.

| Check | Result | What it establishes |
|---|---|---|
| Expanded checkpoint queries, h100 GPU 0 | 13 observations from source 15/30; 78 saved prediction/reference arrays independently checked | Loading, shapes, finite values and training-reference action comparison |
| Native ready return, workstation GPU 1 | COMPLETED after 5.339 s; maximum joint error in the final 0.5 s was 0.04860 rad | Tracking of this candidate from default standing in this provisional scene |
| Grounded simulation screen | All 121 sampled states had foot/floor contact; no virtual support or automatic reset; maximum base tilt 2.66° | Basic standing screen, with 2.31 cm horizontal base displacement |
| Right-hand/forearm geometric screen | Minimum signed distance −0.264 mm at middle fingertip/desk; 33 nonpositive samples | Modeled desk contact/overlap; no passing clearance result |
| Project suite | 709 passed, 19 skipped, 46 warnings | Existing software tests; physical behavior remains outside their scope |
| New scene checks | 6 passed | Preserved joint/actuator addresses, scene inputs and injected signed-distance counterexamples |

The disabled hardware [profile](../../../configs/g1/handover.example.yaml) is
unchanged. The successful ready probe uses a separate simulation fixture with
both readiness gates set true to exercise that component. It uses a scripted
policy only for prewarm and forbids latent-action publication. Those fixture
flags do not qualify a checkpoint or authorize a physical return. The earlier
[unverified attempt](ready-040-gpu1/result.json) was correctly rejected by the
native readiness gate. The [first attempt](ready-040/result.json) failed while
loading the planner on the busy workstation GPU 0; the bounded retry used idle
GPU 1. Every owned child process stopped.

We can estimate reach from the dataset at the grasp moment. The
[grasp frames](grasp-review.jpg) bracket source 15 frames [180,201) and source
30 [168,201). Their stored wrist positions agree with named-joint G1 forward
kinematics within 0.00027 mm. The stored wrist positions are themselves derived
from kinematics; this verifies joint mapping, not independent metrology.

| Grasp bracket | Wrist forward / right from pelvis | Palm mesh centre forward / right from model torso origin |
|---|---|---|
| Source 15 | 22.77 / 15.30 cm | 31.17 / 16.62 cm |
| Source 30 | 21.64 / 11.14 cm | 30.06 / 11.92 cm |

The model torso origin is not a measured point on the chest surface. The palm
mesh centre is also not the bottle centre. These values are useful scene priors;
they leave the hand-to-bottle offset, desk edge and absolute world translation
unknown. Elsewhere in the episodes, stored wrist positions and this FK differ
by up to 3.16 cm, so the agreement above applies only to the checked grasp
brackets. See [all reach values and assumptions](grasp-reach.json).

Distance matters when placing the simulated bottle within reach and checking
whether the hand crosses the desk edge. Diameter changes the finger/contact
geometry. Height and weight affect open-palm support and rolling. The recordings
contain ego RGB and robot state, with no depth stream or bottle-mass field.
They can locate the hand at grasp; recovering a bottle shape from images needs
a calibrated scale/contact model, and weight needs a separate measurement.
These measurements improve simulation acceptance; they are not prerequisites
for loading the VLA or running the existing software harness.

The [finger-opening plot](finger-opening-comparison.png) shows the largest
sampled forecast discrepancy. For source 15 frame 325, the recorded target
opens after 10 future frames, while the prediction opens after 26: a 0.32 s
difference at 50 Hz. Source 30 frame 320 differs by 2 frames, or 0.04 s. These
are saved 40-frame forecasts. They do not establish executed release timing,
because no live policy rollout was run here. Sources 15/30 were training data.

The scene uses the confirmed 85 cm desk and a person proxy across it. The
40 cm desk-front distance, 20 cm × 8 cm / 300 g bottle, desk dimensions and
person position remain provisional. The hand assets explicitly warn of
simulation instability; the proxy has no human arms or hands. The
[distance sweep](clearance-sweep.json) relocates the desk around the same saved
trajectory. It is geometric sensitivity analysis, with no new dynamics trials;
zero distances remain nonpositive/uncertain clearance and do not establish
physical contact by themselves. It cannot select a safe real desk position.

Next work, in order:

1. Fit a sample scene to the grasp-derived hand positions, then establish the
   bottle contact offset, support height and desk edge. Check the full hand and
   forearm path from a visibly empty post-handover pose. The current default
   standing return and near-desk geometry need review before a physical test.
2. Collect stable open-palm offerings with time for vision confirmation. Start
   with a 10–15 s offering pause as a collection target, then measure the actual
   monitor latency. Record the person taking the bottle, an unobstructed empty
   hand and a stationary ready pose. See the [collection checklist](COLLECTION.md).
3. Reserve whole new collection sessions for validation/test before creating
   training windows. The baseline used 43 of 61 source episodes. The confirmed
   successes 15/30 are in training; source 16 is an excluded failure. No confirmed
   unseen successful cycle is available yet.
4. Run a disconnected live VLA/SONIC rollout against the reviewed scene, then
   supervised hardware acceptance before changing either hardware readiness
   flag. New training depends on the reviewed demonstrations and split, not on
   this training-reference comparison alone.

Reproduction uses the existing environments, pinned offline image and original
read-only source assets. Commands and evidence are in [WORK.md](WORK.md) and
[verification.json](verification.json). No physical robot command, source-data
edit, new training run, automatic transfer or dashboard work was performed.
