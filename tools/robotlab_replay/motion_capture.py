"""Optional single-episode OSC diagnostics. No model/physics calls are added.

Install around the existing eval_policy call, AFTER its other runtime wrappers.
Reads simulator data without forward()/update(); preserves original return objects.
Requires NumPy already used by LIBERO; imports neither Torch nor the environment.
"""
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from unittest.mock import patch

import numpy as np

SCHEMA = 'robotlab_motion_v1'


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    out = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(out).all():
        raise ValueError('nonfinite diagnostic value')
    return out


def read_safetensors(path):
    """Bounded numeric stats only; no pickle, Torch, model weights or downloads."""
    path = Path(path)
    if path.stat().st_size > 64 * 1024:
        raise ValueError('statistics file exceeds 64KiB')
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError('invalid safetensors header')
    length = struct.unpack('<Q', raw[:8])[0]
    if length > 32768 or 8 + length > len(raw):
        raise ValueError('invalid safetensors header length')
    header = json.loads(raw[8:8+length]); output = {}; spans=[]
    for name, spec in header.items():
        if name == '__metadata__':
            continue
        dtype = {'F32':'<f4', 'F64':'<f8'}.get(spec['dtype'])
        shape = spec['shape']; start,end = spec['data_offsets']
        if dtype is None or not all(type(n) is int and 0 < n <= 1024 for n in shape):
            raise ValueError('unsupported statistics tensor')
        count = math.prod(shape)
        if not 0 <= start <= end <= len(raw)-8-length or end-start != count*np.dtype(dtype).itemsize:
            raise ValueError('invalid statistics tensor offsets')
        if any(start < b and end > a for a,b in spans):
            raise ValueError('overlapping statistics tensors')
        spans.append((start,end))
        output[name] = np.frombuffer(raw, dtype=dtype, count=count, offset=8+length+start).reshape(shape).copy()
    return output


class ActionDecoder:
    """Exact frozen MEAN_STD output transform: normalized * std + mean."""
    def __init__(self, mean, std, provenance=None):
        self.mean=array(mean); self.std=array(std); self.provenance=provenance or {}
        if self.mean.shape != (7,) or self.std.shape != (7,) or np.any(self.std < 0):
            raise ValueError('requires seven finite mean/std values')

    @classmethod
    def from_snapshot(cls, folder):
        folder=Path(folder).resolve(); config=folder/'policy_postprocessor.json'
        raw=config.read_bytes(); settings=json.loads(raw);steps=settings['steps']
        if [s['registry_name'] for s in steps] != ['unnormalizer_processor','device_processor']:
            raise ValueError('unsupported output processor pipeline')
        spec=steps[0]
        if spec['config']['norm_map']['ACTION'] != 'MEAN_STD' or spec['config']['features']['action']['shape'] != [7]:
            raise ValueError('not frozen seven-dimensional MEAN_STD actions')
        name=spec['state_file']
        if not isinstance(name,str) or Path(name).name!=name or '\\' in name or ':' in name:
            raise ValueError('statistics filename must be a snapshot-local basename')
        logical=folder/name;path=logical.resolve()
        if not path.is_relative_to(folder):
            # HF snapshots intentionally symlink to content-addressed blob stores.
            # Permit only the two recognized cache layouts, never arbitrary links.
            model=folder.parent.parent;hub=model.parent
            roots=[model/'blobs',hub/'blobs'] if folder.parent.name=='snapshots' and model.name.startswith('models--') else []
            if not logical.is_symlink() or not any(path.is_relative_to(root.resolve()) for root in roots):
                raise ValueError('statistics symlink escapes recognized HF blob store')
        values=read_safetensors(path)
        # These are the actual saved LeRobot processor state keys.
        pairs=[(a,b) for a,b in [('action.mean','action.std'),('action_mean','action_std')]
            if a in values and b in values]
        if len(pairs)!=1:
            raise ValueError('unknown/ambiguous action statistics keys: '+str(sorted(values)))
        mean,std=pairs[0]
        return cls(values[mean].reshape(-1),values[std].reshape(-1),
            {'processor_sha256':hashlib.sha256(raw).hexdigest(),
             'statistics_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'formula':'action = normalized * std + mean; no epsilon in inverse',
             'units':'OSC input command, before controller scaling; not joint angles'})

    def decode(self, value):
        v=array(value)
        if v.shape[-1:] != (7,):
            raise ValueError('action final dimension must be seven')
        # Frozen eval uses float32 processor arithmetic.
        return (v.astype(np.float32)*self.std.astype(np.float32)+self.mean.astype(np.float32)).astype(np.float64)


def rotation_increment(v):
    v=array(v);theta=float(np.linalg.norm(v))
    if v.shape!=(3,):raise ValueError('axis-angle must have three dimensions')
    if theta < 1e-12:return np.eye(3)
    x,y,z=v/theta;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(theta)*k+(1-math.cos(theta))*(k@k)


def scaled_delta(action, config):
    if config.get('type')!='OSC_POSE' or config.get('control_delta') is not True:
        raise ValueError('only explicit relative OSC_POSE supported')
    a=array(action)
    if a.shape!=(7,):raise ValueError('expected seven-dimensional command')
    lo=np.broadcast_to(array(config['input_min']), (6,));hi=np.broadcast_to(array(config['input_max']), (6,))
    out_lo=np.broadcast_to(array(config['output_min']), (6,));out_hi=np.broadcast_to(array(config['output_max']), (6,))
    if np.any(hi<=lo):raise ValueError('invalid controller input limits')
    return (np.clip(a[:6],lo,hi)-(lo+hi)/2)*(out_hi-out_lo)/(hi-lo)+(out_hi+out_lo)/2


def nominal_trajectory(position, orientation, commands, config):
    """Perfect tracking interpretation; not measured state or a learned world model."""
    p=array(position);r=array(orientation);result=[]
    if p.shape!=(3,) or r.shape!=(3,3):raise ValueError('invalid initial end-effector pose')
    for action in commands:
        d=scaled_delta(action,config);p=p+d[:3];r=rotation_increment(d[3:])@r
        result.append({'position_m':p.tolist(),'orientation_matrix':r.tolist(),
            'gripper_command':float(action[6])})
    return result


def pose_error(goal, actual):
    rotation=array(goal['orientation_matrix'])@array(actual['eef_orientation_matrix']).T
    return {'position_m':float(np.linalg.norm(array(goal['position_m'])-array(actual['eef_position_m']))),
        'orientation_rad':math.acos(float(np.clip((np.trace(rotation)-1)/2,-1,1)))}


class MotionCapture:
    """Lazy LIBERO binding, one episode per instance, independently bounded output.

    `unwrapped` is the LeRobot LiberoEnv, whose _env is created at reset.
    Samples every control transition; torque summaries come from real controller
    calls. Optional substep samples are bounded, not an extra physics rollout.
    """
    def __init__(self, unwrapped, policy, output, decoder, *, method, provenance,
                 fps=20, max_bytes=32*2**20, substep_stride=0):
        if fps!=20 or type(substep_stride) is not int or substep_stride<0:
            raise ValueError('requires explicit 20Hz, nonnegative substep sampling stride')
        if not 1024 <= max_bytes <= 128*2**20:raise ValueError('capture budget must be 1KiB..128MiB')
        self.env=unwrapped;self.policy=policy;self.output=Path(output);self.decoder=decoder
        self.method=method;self.provenance=provenance;self.fps=fps;self.budget=max_bytes
        self.substep_stride=substep_stride;self.stack=ExitStack();self.bytes=0;self.steps=0
        self.resets=0;self.in_reset=False;self.active=None;self.chunk=None;self.slot=0;self.selected=None
        self.chunks=0;self.overhead_s=0.;self.bound=False;self.raw=None

    def _json(self, name, value):
        raw=(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n').encode()
        if self.bytes+len(raw)>self.budget:raise RuntimeError('motion capture byte budget exceeded')
        p=self.output/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(raw)
        self.bytes+=len(raw)

    def _state(self):
        data=self.raw.sim.data;robot=self.robot
        state={'sim_time_s':float(data.time),
            'eef_position_m':array(data.site_xpos[robot.eef_site_id]).tolist(),
            'eef_orientation_matrix':array(data.site_xmat[robot.eef_site_id]).reshape(3,3).tolist(),
            'joint_position_rad':array(data.qpos[robot._ref_joint_pos_indexes]).tolist(),
            'joint_velocity_rad_s':array(data.qvel[robot._ref_joint_vel_indexes]).tolist(),
            'gripper_position':array(data.qpos[robot._ref_gripper_joint_pos_indexes]).tolist(),
            'gripper_velocity':array(data.qvel[robot._ref_gripper_joint_vel_indexes]).tolist(),
            'contact_count_all_scene':int(data.ncon)}
        if len(state['joint_position_rad'])!=7:raise ValueError('expected Panda seven arm joints')
        if hasattr(data,'actuator_force'):
            state['actuator_force_last_dynamics']=array(data.actuator_force[robot._ref_joint_actuator_indexes]).tolist()
        if hasattr(data,'qfrc_actuator'):
            state['joint_actuator_force_last_dynamics']=array(data.qfrc_actuator[robot._ref_joint_vel_indexes]).tolist()
        return state

    def _bind(self):
        self.raw=self.env._env
        if self.raw is None or len(self.raw.robots)!=1:raise ValueError('requires initialized single-robot LIBERO')
        self.robot=self.raw.robots[0];self.controller=self.robot.controller;c=self.controller
        if c.name!='OSC_POSE' or c.use_delta is not True or c.control_dim!=6:
            raise ValueError('requires actual relative six-dimensional OSC_POSE controller')
        self.config={k:array(getattr(c,k)).tolist() for k in ['input_min','input_max','output_min','output_max','kp','kd']}
        self.config.update(type=c.name,control_delta=bool(c.use_delta))
        initial=self._state()
        self._json('protocol.json',{'schema':SCHEMA,'method':self.method,'provenance':self.provenance,
            'decoder':self.decoder.provenance,'controller':self.config,'control_fps':self.fps,
            'initial_state_after_reset_settling':initial,'substep_stride':self.substep_stride,
            'state_read':'direct sim.data; no controller.update or sim.forward added',
            'video_alignment':'before state of transition n is Z_n; final after state Z_N may not be in video',
            'chunk_semantics':'returned final policy action chunk; guidance included when active; no unguided counterfactual computed',
            'nominal_semantics':'block-anchored perfect tracking interpretation; not a physical prediction',
            'torque_semantics':'controller output, post-limit command, and last evaluated MuJoCo actuator forces are distinct',
            'joint_targets':'no policy joint-angle target; OSC task-space goal and nullspace reference only',
            'nullspace_reference_joint_position_rad':array(c.initial_joint).tolist() if hasattr(c,'initial_joint') else None,
            'torque_limits':array(self.robot.torque_limits).tolist() if hasattr(self.robot,'torque_limits') else None,
            'timing_limit':'CPU copies and IO add wall time; no explicit CUDA synchronize, but device-to-host copies may synchronize GPU'})
        original_step=self.raw.step;original_goal=c.set_goal;original_run=c.run_controller;original_control=self.robot.control
        def goal(*a,**kw):
            value=original_goal(*a,**kw)
            if self.active is not None:
                self.active['goals'].append({'sim_time_s':float(self.raw.sim.data.time),
                    'position_m':array(c.goal_pos).tolist(),'orientation_matrix':array(c.goal_ori).tolist()})
            return value
        def run(*a,**kw):
            value=original_run(*a,**kw)
            if self.active is not None:self.active['last_unclipped']=array(value).tolist()
            return value
        def control(*a,**kw):
            value=original_control(*a,**kw)
            if self.active is not None:
                row=self.active;raw=array(row['last_unclipped']);command=array(self.robot.torques)
                row['controller_updates']+=1;row['clipped_updates']+=int(not np.array_equal(raw,command))
                sample={'sim_time_s':float(self.raw.sim.data.time),'pre_limit_torque':raw.tolist(),
                    'post_limit_torque':command.tolist(),
                    'actuator_ctrl':array(self.raw.sim.data.ctrl[self.robot._ref_joint_actuator_indexes]).tolist()}
                row.setdefault('first_torque',sample);row['last_torque']=sample
                row['peak_command_torque_l2']=max(row['peak_command_torque_l2'],float(np.linalg.norm(command)))
                if self.substep_stride and (row['controller_updates']-1)%self.substep_stride==0:
                    if len(row['substeps'])>=1000:raise RuntimeError('substep budget exceeded')
                    row['substeps'].append(sample)
            return value
        def step(action,*a,**kw):
            if self.in_reset:return original_step(action,*a,**kw)
            if self.selected is None:raise RuntimeError('control command has no corresponding policy selection')
            command=array(action)
            if command.shape!=(7,) or not np.allclose(command,self.decoder.decode(self.selected),rtol=2e-5,atol=2e-6):
                raise ValueError('captured action decoder differs from actual submitted command')
            before=self._state();row={'step':self.steps,'chunk_id':self.chunks,'slot':self.slot-1,
                'selected_normalized_action':self.selected.tolist(),'submitted_command':command.tolist(),
                'before':before,'goals':[],'controller_updates':0,'clipped_updates':0,
                'peak_command_torque_l2':0.,'substeps':[]};self.active=row
            started=time.perf_counter()
            try:value=original_step(action,*a,**kw)
            finally:self.active=None
            elapsed=time.perf_counter()-started;began=time.perf_counter();after=self._state()
            if len(row['goals'])!=1 or row['controller_updates']==0:
                raise RuntimeError('expected one real set_goal and at least one controller update per action')
            dt=after['sim_time_s']-before['sim_time_s']
            if not math.isclose(dt,1/self.fps,rel_tol=1e-5,abs_tol=1e-7):
                raise RuntimeError('observed simulation/control timebase differs')
            row.pop('last_unclipped',None);row.update(after=after,returned=True,
                tracking_error=pose_error(row['goals'][0],after),
                nominal_block_error=pose_error(self.chunk['nominal'][self.slot-1],after),
                original_control_step_wall_s=elapsed)
            self.overhead_s+=time.perf_counter()-began
            raw=(json.dumps(row,allow_nan=False)+'\n').encode()
            if self.bytes+len(raw)>self.budget:raise RuntimeError('motion capture byte budget exceeded')
            self.stream.write(raw);self.stream.flush();self.bytes+=len(raw)
            self.steps+=1;self.selected=None
            return value
        for obj,name,fn in [(self.raw,'step',step),(c,'set_goal',goal),(c,'run_controller',run),(self.robot,'control',control)]:
            self.stack.enter_context(patch.object(obj,name,fn))
        self.bound=True

    def __enter__(self):
        self.output.mkdir(parents=True,exist_ok=False)
        self.stream=(self.output/'steps.jsonl').open('xb')
        original_reset=self.env.reset;original_chunk=self.policy._get_action_chunk;original_select=self.policy.select_action
        def reset(*a,**kw):
            if self.resets:raise RuntimeError('one episode per motion capture; create a new attempt for another reset')
            self.in_reset=True
            try:value=original_reset(*a,**kw)
            finally:self.in_reset=False
            self.resets+=1;self._bind();return value
        def chunk(*a,**kw):
            if not self.bound:raise RuntimeError('motion capture requires completed reset before policy inference')
            origin=self._state();value=original_chunk(*a,**kw);began=time.perf_counter();v=array(value)
            if v.shape!=(1,50,7):raise ValueError('requires actual complete [1,50,7] returned chunk')
            commands=self.decoder.decode(v[0]);self.chunks+=1;self.slot=0
            self.chunk={'chunk_id':self.chunks,'origin_step':self.steps,'origin':origin,
                'normalized_actions':v[0].tolist(),'decoded_commands':commands.tolist(),
                'nominal':nominal_trajectory(origin['eef_position_m'],origin['eef_orientation_matrix'],commands,self.config),
                'role':'final_policy_chunk_with_possible_guidance' if self.method.endswith('ogg') else 'unguided_returned_policy_chunk'}
            self._json(f'chunks/{self.chunks:04d}.json',self.chunk);self.overhead_s+=time.perf_counter()-began
            return value
        def select(*a,**kw):
            value=original_select(*a,**kw);began=time.perf_counter();v=array(value).reshape(-1)
            if self.selected is not None or self.chunk is None or self.slot>=50:
                raise RuntimeError('selection without new chunk, or prior action not submitted')
            if v.shape!=(7,) or not np.array_equal(v,np.array(self.chunk['normalized_actions'][self.slot])):
                raise ValueError('selected action does not match actual chunk/slot')
            self.selected=v;self.slot+=1;self.overhead_s+=time.perf_counter()-began
            return value
        try:
            for obj,name,fn in [(self.env,'reset',reset),(self.policy,'_get_action_chunk',chunk),(self.policy,'select_action',select)]:
                self.stack.enter_context(patch.object(obj,name,fn))
        except BaseException:
            self.stack.close();self.stream.close();raise
        return self

    def __exit__(self,typ,error,tb):
        self.stack.close();self.stream.close()
        complete=typ is None and self.steps>0 and self.selected is None
        status={'schema':SCHEMA,'status':'completed' if complete else 'interrupted_or_error',
            'steps':self.steps,'chunks':self.chunks,'bytes':self.bytes,'python_capture_copy_overhead_s':self.overhead_s,
            'overhead_limit':'partial instrumentation cost only; no claim of zero latency or benchmark overhead',
            'error':None if error is None else type(error).__name__+': '+str(error),
            'source_changed':False,'extra_policy_forwards':0,'extra_physics_steps':0}
        (self.output/'status.json').write_text(json.dumps(status,ensure_ascii=False,allow_nan=False)+'\n')
        if typ is None and not complete:raise RuntimeError('motion episode did not complete all selected submissions')
        return False
