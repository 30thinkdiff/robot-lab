"""CPU proof fixtures for lossless feature transport; GPU parity tested separately."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.utils.data import ConcatDataset,Subset,DataLoader
from torch.utils.data._utils.collate import default_collate

from . import data
from .packed_pipeline import PackedEpisode,packed_groups,packed_move,loader_override
from .test_core import fixture,reference,plan


class PackedTests(unittest.TestCase):
    def tearDown(self):data.POOL.close()

    def test_real_shape_fixture_inputs_targets_and_action_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            spec=fixture(Path(temp));original=data.EpisodeDataset(spec,'block_02')
            packed=PackedEpisode(original)
            expected=default_collate([original[0]])
            actual=packed_move(default_collate([packed[0]]),'cpu')
            self.assertEqual(set(actual),{'z_t','delta_z','a_t'})
            for key in actual:
                self.assertEqual(actual[key].dtype,expected[key].dtype)
                self.assertTrue(torch.equal(actual[key].view(torch.int32),expected[key].view(torch.int32)))
            # A changed target frame cannot affect either model condition.
            data.POOL.close()
            p=Path(spec['raw_path'])/'block_02.fp16.npy';raw=np.load(p,mmap_mode='r+')
            raw[12]=7;raw.flush();raw._mmap.close()
            changed=packed_move(default_collate([packed[0]]),'cpu')
            self.assertTrue(torch.equal(changed['z_t'],actual['z_t']))
            self.assertTrue(torch.equal(changed['a_t'],actual['a_t']))
            self.assertFalse(torch.equal(changed['delta_z'],actual['delta_z']))
            group=packed_groups({'train':ConcatDataset([original])})
            self.assertEqual(len(group['train']),len(original))

    def test_extreme_half_values_convert_before_subtraction_and_do_not_leak(self):
        values=np.arange(65536,dtype=np.uint16).view(np.float16)
        values=values[np.isfinite(values)].copy()
        z=torch.from_numpy(values).reshape(1,-1,1)
        future=z.flip(1).contiguous()
        batch=dict(packed_current=z,packed_future=future,a_t=torch.zeros(1,7),sample_index=torch.tensor([0]))
        actual=packed_move(batch,'cpu');expected=future.float()-z.float()
        self.assertTrue(torch.equal(actual['delta_z'].view(torch.int32),expected.view(torch.int32)))
        self.assertEqual(set(actual),{'z_t','delta_z','a_t'})
        self.assertTrue(torch.isfinite(actual['delta_z']).all())
        with self.assertRaises(ValueError):packed_move(batch,'cpu',history=True)
        with self.assertRaises(ValueError):packed_move(dict(batch,unexpected=torch.ones(1)),'cpu')

    def test_frozen_subset_split_and_full_index_population_preserved(self):
        ref=reference();p=plan()
        groups={s:ConcatDataset([data.EpisodeDataset(r,'block_02') for r in p['records'] if r['split']==s])
                for s in ('train','validation')}
        split,info,indices=ref['train'].select_split(groups,'episode',.1,42)
        packed=packed_groups(split)
        self.assertEqual(info['samples'],{'train':49447,'validation':12089})
        for part in split:
            self.assertIsInstance(packed[part],Subset)
            self.assertIs(packed[part].indices,split[part].indices)
            self.assertEqual(packed[part].dataset.cumulative_sizes,split[part].dataset.cumulative_sizes)
            for original,wrapped in zip(split[part].dataset.datasets,packed[part].dataset.datasets):
                self.assertIs(wrapped.original,original)
                self.assertEqual(original.pairs,wrapped.original.pairs)
        self.assertIs(packed['train'].dataset,packed['validation'].dataset)
        with tempfile.TemporaryDirectory() as temp:
            spec=fixture(Path(temp));original=data.EpisodeDataset(spec,'block_02')
            subset=Subset(ConcatDataset([original]),[0]);gen=torch.Generator().manual_seed(42)
            with loader_override(ref,4):
                ls=ref['train'].reference_loaders({'train':subset,'validation':subset},32,gen,num_workers=4,pin_memory=False)
                actual=packed_move(next(iter(ls[0])),'cpu')
            expected=default_collate([original[0]])
            for key in actual:self.assertTrue(torch.equal(actual[key],expected[key]))


if __name__=='__main__':unittest.main()
