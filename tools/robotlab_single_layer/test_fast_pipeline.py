"""CPU fixture checks; real full-resolution GPU parity is a separate explicit pilot."""
import copy
import tempfile
from pathlib import Path
import unittest

import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.data._utils.collate import default_collate

from .fast_pipeline import mlp_collate,tensor_digest
from .fast_migration import migrate
from .train_monitor import write


class OrderFixture(Dataset):
    def __len__(self):return 49447
    def __getitem__(self,index):
        z=torch.tensor([[float(index)]])
        return dict(z_t=z,z_hist=z.unsqueeze(0),a_t=torch.zeros(7),delta_z=z*0,sample_index=index)


class FastPipelineTests(unittest.TestCase):
    def test_collate_retained_fields_exact_and_rejects_multihistory(self):
        samples=[OrderFixture()[i] for i in range(32)]
        old=default_collate(samples);new=mlp_collate(samples)
        self.assertNotIn('z_hist',new)
        for key in new:self.assertTrue(torch.equal(old[key],new[key]))
        samples[0]['z_hist']=samples[0]['z_hist'].repeat(3,1,1)
        with self.assertRaises(ValueError):mlp_collate(samples)

    def test_full_epoch_order_and_generator_state_across_workers(self):
        records=[]
        for workers,collate in [(2,default_collate),(4,mlp_collate),(8,mlp_collate)]:
            generator=torch.Generator().manual_seed(42)
            torch.manual_seed(53)
            loader=DataLoader(OrderFixture(),batch_size=32,shuffle=True,generator=generator,
                              num_workers=workers,collate_fn=collate)
            indices=[]
            for batch in loader:indices.extend(batch['sample_index'].tolist())
            order=torch.tensor(indices,dtype=torch.int64)
            records.append((tensor_digest(order),tensor_digest(generator.get_state()),tensor_digest(torch.get_rng_state())))
        self.assertEqual(records[0],records[1]);self.assertEqual(records[0],records[2])

    def test_migration_changes_only_engineering_protocol_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/'old';src.mkdir();dst=root/'new'
            protocol=dict(model_type='mlp',h_window=1,batch_size=32,accumulate=16,loader=dict(num_workers=2,pin_memory=True))
            payload=dict(protocol=protocol,epoch=7,step=679,state_dict={'weight':torch.randn(2,3)},
                optimizer={'moments':torch.randn(2,3)},scheduler={'last_epoch':7},rng=torch.get_rng_state(),
                loader_rng=torch.Generator().manual_seed(42).get_state(),history=[{'epoch':7}],best=1.,stale=0)
            torch.save(payload,src/'latest.pt');write(src/'protocol.json',protocol)
            write(src/'training_status.json',dict(status='paused',epochs_completed=7))
            record=migrate(src,dst,4,root/'accepted.json')
            self.assertTrue(record['checkpoint_payload_exact']);self.assertEqual(record['inherited_epoch'],7)
            check=torch.load(src/'latest.pt',weights_only=True);self.assertEqual(check['protocol']['loader']['num_workers'],2)
            new=torch.load(dst/'latest.pt',weights_only=True);self.assertEqual(new['protocol']['loader']['num_workers'],4)
            self.assertTrue(torch.equal(check['loader_rng'],new['loader_rng']))
            packed=root/'packed'
            receipt=migrate(src,packed,8,root/'packed_accepted.json',pipeline_kind='packed_fp16_transport')
            packed_state=torch.load(packed/'latest.pt',weights_only=True)
            self.assertEqual(receipt['changes']['pipeline_kind'],'packed_fp16_transport')
            self.assertEqual(receipt['checkpoint_payload_except_protocol_sha256'],record['checkpoint_payload_except_protocol_sha256'])
            self.assertTrue(torch.equal(check['loader_rng'],packed_state['loader_rng']))


if __name__=='__main__':unittest.main()
