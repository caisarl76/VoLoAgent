"""Read-only native target mapping probe, no publisher instantiated."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

OUT=Path(__file__).parent
G=Path('/home/jihun/work/VoLoAgent/.worktrees/g1-harness-native')
sys.path.insert(0,str(G))
from gear_sonic.utils.inference.standing_reset import StandingReset
from gear_sonic.utils.teleop.xr_upperbody_bridge import G1_UPPER_BODY_JOINT_INDICES

z=np.load(OUT/'measured-extract.npz')
names=z['names'].tolist()
q=z['15/observation.state'][415]
motors=[x.attrib['joint'] for x in ET.parse(G/'gear_sonic_deploy/g1/g1_29dof_with_hand.xml').getroot().findall('actuator/motor')][:29]
upper=[motors[i] for i in G1_UPPER_BODY_JOINT_INDICES]
right=motors[22:29]
ready=json.loads((OUT.parents[1]/'g1_handover_20261009/ready-pose-candidate.json').read_text())['joints_radians']
# Hands are held unchanged dummy vectors; no hand-hardware-order claim needed.
state={'body_q_measured_motor':[q[names.index(n)] for n in motors],
       'base_quat':z['15/observation.root_orientation'][415].tolist(),
       'left_hand_q_measured':[0.]*7,'right_hand_q_measured':[0.]*7}
reset=StandingReset(state,[0.]*7,[0.]*7,right_arm_target=[ready[n] for n in right])
expected=np.array([ready[n] if n in right else q[names.index(n)] for n in upper])
actual=reset.target[:17]
rows=[{'planner_index':i,'joint':n,'measured':float(q[names.index(n)]),'expected':float(expected[i]),'actual':float(actual[i]),'error_rad':float(actual[i]-expected[i])} for i,n in enumerate(upper)]
result={'source_sha256':hashlib.sha256((G/'gear_sonic/utils/inference/standing_reset.py').read_bytes()).hexdigest(),
 'mapping_pass':bool(np.max(np.abs(actual-expected))<1e-7),
 'max_target_mapping_error_rad':float(np.max(np.abs(actual-expected))),
 'preserved_non_right_arm':bool(all(abs(r['actual']-r['measured'])<1e-7 for r in rows if r['joint'] not in right)),
 'planner_upperbody_order':upper,'rows':rows,'hands':'Unchanged zero fixtures only, no hand motor-order conversion attempted.',
 'control_connection_opened':False}
(OUT/'native-ready-mapping-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
