"""Tests use only generated fictional inputs, never experimental measurements."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
from script import data, plotting
from script.synthetic_demo import generate


class CodeOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=generate(Path(self.temp.name)/'fictional')
        self.patches=[patch.object(data,'DATA_DIR',self.root/'data'),
                      patch.object(plotting,'DATA_DIR',self.root/'data')]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in self.patches:p.stop()
        self.temp.cleanup()

    def test_schema_and_systems(self):
        self.assertEqual(len(data.SYSTEMS),6)
        for s in data.SYSTEMS:
            f,p=data.load_preos_predictions(s)
            self.assertEqual(len(f),54)
            self.assertTrue(np.isfinite(f[p]).all())
        self.assertTrue((self.root/'data/SYNTHETIC_DEMO.json').exists())

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):generate(self.root)

    def test_plot(self):
        f=plotting.plot_system('C5N2',self.root/'proposed_predictions.csv',self.root/'plots')
        self.assertGreater(f.stat().st_size,1000)
        self.assertTrue(f.with_suffix('.png').exists())

    def test_training(self):
        from script import shared_model as model
        model.set_seed(42)
        raw,stats,_=model.load_target_dataframe('dP')
        _,pred,meta,_=model.train_model(raw,'dP',1,.005,1e-6,12,42,.2)
        self.assertEqual(len(pred),324)
        self.assertTrue(np.isfinite(model.denormalize_predictions(pred,stats,'dP').P_pred_nn).all())

    def test_numerical_root(self):
        from script import pr76
        with patch.object(pr76,'PROPS_FILE',self.root/'data/properties.csv'):
            props=pr76.get_system_props('C5N2')
        self.assertTrue(all(np.isfinite(pr76.pr76_pure_params(350.,props['Tc'],props['Pc'],props['omega']))))

    def test_missing_input_message(self):
        with patch.object(data,'DATA_DIR',self.root/'missing'):
            with self.assertRaisesRegex(FileNotFoundError,'Supply your own data'):
                data.load_preos_predictions('C5N2')


if __name__=='__main__':unittest.main()
