"""Selection isolation and degenerate-family handling, separate baseline runtime."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import cspi_supported_candidates as adapter


def data(lo=-.77, hi=.77):
    t=np.arange(32)%2
    return pd.DataFrame(dict(treatment=t, propensity=np.full(32,.5),
        outcome=t*(1+np.arange(32)/20), mu_0_est=np.zeros(32), mu_1_est=np.ones(32),
        ite_est=np.linspace(lo,hi,32)))


class SupportedCandidates(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        def pde(cutoffs, baseline, frame, by_quantile=False, dr=True):
            x,_=adapter.influence(frame,cutoffs,dr)
            return x.mean(axis=1),np.cov(x)/x.shape[1]
        def choose(cutoffs, baseline, frame, **kwargs):
            self.calls.append(('choose',cutoffs.tolist()))
            return np.repeat(cutoffs[0],2),np.repeat(cutoffs[0],2),cutoffs[:1],cutoffs[:2]
        def test(cutoffs, baseline, frame, **kwargs):
            self.calls.append(('test',cutoffs.tolist()))
            return np.ones(len(cutoffs),bool)
        self.f=dict(PDE=pde,cutoff_selection=choose,safety_test_single=test,safety_test=test)
        self.fit=np.linspace(-1,1,64)

    def test_tuning_filter_removes_only_uncomputable_candidate(self):
        chosen=adapter.choose(self.f,self.fit,data(),1)['payload']
        self.assertEqual(len(chosen['retained']),4)
        self.assertEqual(chosen['diagnostics'][0]['support'][-1],0)
        self.assertEqual(self.calls[0][1],chosen['retained'])

    def test_negative_gain_candidates_are_not_removed(self):
        frame=data();frame['outcome']=-frame.outcome;frame['mu_1_est']=-1.
        chosen=adapter.choose(self.f,self.fit,frame,1)['payload']
        self.assertEqual(len(chosen['retained']),4)
        self.assertTrue(all(v<0 for v in chosen['diagnostics'][1]['estimates'][:4]))

    def test_heldout_reward_changes_cannot_reselect_or_mutate_selection(self):
        selection=adapter.choose(self.f,self.fit,data(),1)
        before=deepcopy(selection)
        first=adapter.test_selected(self.f,selection,data(-.76,.76),2)
        altered=data(-.76,.76);altered.outcome+=1e6
        second=adapter.test_selected(self.f,selection,altered,2)
        self.assertEqual(selection,before)
        self.assertEqual(first['selection_sha256'],second['selection_sha256'])
        self.assertEqual(sum(c[0]=='choose' for c in self.calls),1)

    def test_test_degeneracy_rejects_family_without_search(self):
        selection=adapter.choose(self.f,self.fit,data(),1)
        result=adapter.test_selected(self.f,selection,data(-2,-1.1),2)
        self.assertTrue(all(m['status']=='untestable' for m in result['methods']))
        self.assertTrue(all(m['deployed_cutoff']==adapter.BASELINE for m in result['methods']))
        self.assertEqual([c[0] for c in self.calls],['choose'])

    def test_one_degenerate_member_rejects_entire_multiple_family(self):
        selection=adapter.choose(self.f,self.fit,data(),1)
        result=adapter.test_selected(self.f,selection,data(-.7,-.6),2)
        self.assertEqual(result['methods'][0]['status'],'tested')
        self.assertEqual(result['methods'][1]['status'],'untestable')
        self.assertEqual(result['methods'][1]['selected'],selection['payload']['selected']['cspi_mt'])
        self.assertEqual(sum(c[0]=='test' for c in self.calls),1)

    def test_singleton_goes_directly_to_author_test(self):
        selection=adapter.choose(self.f,self.fit,data(-.7,-.6),1)
        self.assertEqual(selection['payload']['route'],'singleton_no_search')
        adapter.test_selected(self.f,selection,data(-.69,-.61),2)
        self.assertEqual([c[0] for c in self.calls],['test','test'])

    def test_no_support_never_calls_author_selector_or_test(self):
        selection=adapter.choose(self.f,self.fit,data(-2,-1.1),1)
        result=adapter.test_selected(self.f,selection,data(),2)
        self.assertEqual(self.calls,[])
        self.assertTrue(all(m['status']=='untestable' for m in result['methods']))

    def test_mutated_selection_is_rejected(self):
        selection=adapter.choose(self.f,self.fit,data(),1)
        selection['payload']['selected']['cspi']=[-.123]
        with self.assertRaisesRegex(ValueError,'changed'):
            adapter.test_selected(self.f,selection,data(),2)

    def test_endpoint_and_no_overlap_are_rejected(self):
        frame=data();frame.loc[0,'ite_est']=0.
        with self.assertRaisesRegex(ValueError,'endpoint'):
            adapter.influence(frame,np.array([0.]),True)
        frame=data();frame.loc[0,'propensity']=1.
        with self.assertRaisesRegex(ValueError,'overlap'):
            adapter.influence(frame,np.array([0.]),True)

    def test_duplicate_training_candidates_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            adapter.choose(self.f,np.zeros(64),data(),1)


if __name__=='__main__':
    unittest.main()
