"""Outcome-free full-size schedule;registering/testing is not alpha allocation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import matched_plant_parameters
from coopgate.episode_thinning_control import independent_coins
from mappo_formal_common import planned_initial,array_sha256,canonical_sha256
from mappo_coupled_formal_runtime import CALIBRATION_METHODS,CONFIRMATION_METHODS

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101'
SCENES=('nominal','dense','slow_yield','asymmetric_speed','tight_headway')
PLANTS=('nominal','gain_drag','lag_mild','lag_strong');MODES=('clean','uniform','bias','pulse')
COUNTS=dict(calibration=3072,confirmation=1024)
SEEDS=dict(calibration=862000001,confirmation=863000001)
STREAMS=dict(calibration=540101,confirmation=540201)
COIN_STREAM=540301


def stage_cases(stage,probabilities=None):
    if stage not in COUNTS:raise ValueError('Unknown full formal stage')
    n=COUNTS[stage];rng=np.random.default_rng(np.random.SeedSequence([STREAMS[stage],n]))
    scenes=rng.integers(5,size=n);plants=np.arange(n)%4 if stage=='calibration' else rng.integers(4,size=n)
    modes=rng.integers(4,size=n);methods=CALIBRATION_METHODS if stage=='calibration' else CONFIRMATION_METHODS
    coins={}
    if stage=='confirmation' and probabilities is not None:
        for i,seed in enumerate(('4703','4704','4705')):
            value=probabilities[seed]
            coins[seed]=(np.zeros(n,dtype=bool) if value is None else independent_coins(value,n,stream=COIN_STREAM,model_index=i,stage=1))
    result=[]
    for index in range(n):
        seed=SEEDS[stage]+index;scene=SCENES[scenes[index]];plant=PLANTS[plants[index]];mode=MODES[modes[index]]
        offset=index%len(methods);order=methods[offset:]+methods[:offset]
        if index%2:order=order[::-1]
        case=dict(stage=stage,index=index,cluster_id=f'mappo4:{stage}:{index:04d}',seed=seed,scenario=scene,plant=plant,disturbance=mode,
            methods=list(order),initial_sha256=array_sha256(planned_initial(seed,scene),decimals=8),
            dynamics_sha256=canonical_sha256(matched_plant_parameters(seed,plant,.1)),
            disturbance_sha256=array_sha256(disturbance_tape(seed,mode,.1,250)))
        if coins:case['controller_coins']={s:bool(v[index]) for s,v in coins.items()}
        result.append(case)
    if stage=='calibration' and Counter(c['plant'] for c in result)!={p:768 for p in PLANTS}:raise ValueError('Calibration strata changed')
    return result


def validate_candidate():
    path=BASE/'compact_coupled_fixed_candidate_20261102.json';candidate=json.loads(path.read_text())
    if candidate['common_gain_floor']!=.0025 or candidate['actor_seeds']!=[4703,4704,4705]:raise ValueError('Candidate changed')
    for name,digest in candidate['controller_source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen controller changed: '+name)
    bank=BASE/"development/compact_coupled_bank_20261102.json"
    if candidate["controller"]!="CompactCoupledGate" or hashlib.sha256(bank.read_bytes()).hexdigest()!=candidate["coupled_bank_sha256"]:raise ValueError("Coupled bank changed")
    return candidate


def validation_cases():
    result=[]
    for index,plant in enumerate(PLANTS):
        seed=866000001+index;scene=SCENES[index];mode=MODES[index]
        result.append(dict(stage='validation_not_formal',index=index,cluster_id=f'mappo4:validation:{index}',seed=seed,
            scenario=scene,plant=plant,disturbance=mode,methods=list(CONFIRMATION_METHODS),
            controller_coins={s:bool(independent_coins(.5,4,stream=COIN_STREAM,model_index=i,stage=99)[index])
                for i,s in enumerate(('4703','4704','4705'))},
            initial_sha256=array_sha256(planned_initial(seed,scene),decimals=8),
            dynamics_sha256=canonical_sha256(matched_plant_parameters(seed,plant,.1)),
            disturbance_sha256=array_sha256(disturbance_tape(seed,mode,.1,250))))
    return result
