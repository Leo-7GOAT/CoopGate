"""Append fixed development-selected coupled witnesses to the unchanged gate.

Same genuine MAPPO actors, nominal ordering, physical baseline and 0.0025 floor.
One-to-three additional paths per uncertain class; all original paths remain.
This is a new unvalidated development controller, not a deployment certificate.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .pruned_gain_margin_gate import PrunedGainMarginGate
from .coupled_corner_witness import coupled_library


class CompactCoupledGate(PrunedGainMarginGate):
    def __init__(self,model,profile,bank_path):
        bank_path=Path(bank_path)
        raw=bank_path.read_bytes();bank=json.loads(raw)
        if not bank['development_data_only'] or bank['alpha_allocated']:
            raise ValueError('Expected explicit development-only fixed bank')
        super().__init__(model,profile,.0025)
        self.coupled_bank_sha256=hashlib.sha256(raw).hexdigest()
        paths=bank['profiles'][profile]['paths'];library=list(coupled_library(profile))
        self.coupled_ids=[]
        for path in paths:
            gain,drag,rho,bias=library[path['path_id']]
            if not all(np.array_equal(a,path[b]) for a,b in ((gain,'gain'),(drag,'drag'),(rho,'rho'),(bias,'bias'))):
                raise ValueError('Selected witness differs from declared-class library')
            self.parameter_scenarios.append((gain,drag,rho))
            self.parameter_memories=np.concatenate((self.parameter_memories,np.zeros((1,3))))
            self.tapes=np.concatenate((self.tapes,np.tile(bias,(1,250,1))))
            self.names=tuple(self.names)+(f'coupled_bias_vertex:{path["path_id"]}',)
            self.coupled_ids.append(path['path_id'])

    def select(self,state,previous,unused=None):
        command,info=super().select(state,previous,unused)
        info.update(development_controller='compact_coupled_bias_vertices',
            coupled_bank_sha256=self.coupled_bank_sha256,coupled_path_ids=list(self.coupled_ids),
            coupled_path_count=len(self.coupled_ids),full_parameter_box_certified=False)
        return command,info
