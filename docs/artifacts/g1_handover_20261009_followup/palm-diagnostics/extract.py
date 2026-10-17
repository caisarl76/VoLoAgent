"""Read-only source extraction; run with the existing Isaac-GR00T pyarrow env."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq

OUT = Path(__file__).parent
SRC = Path('/home/jihun/work/GR00T-WholeBodyControl/outputs/handover_bottle_260930')
info = json.loads((SRC / 'meta/info.json').read_text())
hashes = {}
arrays = {'names': np.array(info['features']['observation.state']['names'])}
for ep in (15, 16, 30):
    p = SRC / info['data_path'].format(episode_chunk=0, episode_index=ep)
    v = SRC / info['video_path'].format(episode_chunk=0, episode_index=ep, video_key='observation.images.ego_view')
    hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    hashes[str(v)] = hashlib.sha256(v.read_bytes()).hexdigest()
    t = pq.read_table(p)
    for key in ('observation.state', 'observation.root_orientation', 'observation.projected_gravity', 'observation.eef_state', 'frame_index', 'timestamp'):
        arrays[f'{ep}/{key}'] = np.array(t[key].to_pylist())
    assert np.array_equal(arrays[f'{ep}/frame_index'], np.arange(len(t)))
for name in ('info', 'modality'):
    p = SRC / f'meta/{name}.json'
    hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
np.savez_compressed(OUT / 'measured-extract.npz', **arrays)
(OUT / 'source-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
print({ep: arrays[f'{ep}/observation.state'].shape for ep in (15,16,30)})
