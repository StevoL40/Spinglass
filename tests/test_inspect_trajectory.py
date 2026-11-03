"""Execute notebook code against synthetic archives without a Jupyter kernel."""
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('kind', ['ensemble', 'simulation', 'finite_difference', 'legacy'])
def test_notebook_plots_all_runs_and_preserves_old_formats(tmp_path, monkeypatch, kind):
    matplotlib=pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from simulate_spinglass import Parameters, run_ensemble
    notebook=json.loads((Path(__file__).resolve().parents[1]/'inspect_trajectory.ipynb').read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt,'show',lambda:None)
    if kind in ('ensemble','simulation'):
        p=Parameters(n_spins=20,t_end=.3,record_interval=.1,n_runs=4 if kind=='ensemble' else 1)
        result=run_ensemble(p)
        result['parameters']=json.dumps({'n_runs':p.n_runs})
    else:
        result=dict(trajectory=np.array([[.1,.2,.3,1.,1e-7],[.2,.1,.4,1.,1e-7]]),
                    rho=np.array([[.1,.3],[.2,.4]]),h=np.array([-1.,1.]),parameters='{}')
        if kind=='finite_difference': result['energy']=np.array([-.2,-.3])
    archive=tmp_path/'input.npz'
    np.savez_compressed(archive,**result)
    namespace={}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    source=''.join(cell['source']).replace('filename = None', f'filename = Path({str(archive)!r})')
                    exec(compile(source,'inspect_trajectory.ipynb','exec'),namespace)
        axes=namespace['axes']
        if kind=='ensemble':
            for axis,column in zip(axes,[1,2]):
                assert len(axis.lines)==4
                for index,line in enumerate(axis.lines):
                    assert line.get_linewidth()==pytest.approx(max(0.3, 2.0/4))
                    np.testing.assert_array_equal(line.get_ydata(),result['trajectories'][index,:,column])
                    assert line.get_marker()=='None'
        else:
            assert len(axes)==(2 if kind=='simulation' else 3)
            assert all(len(ax.lines)==1 for ax in axes)
        assert (tmp_path/'figures/input_trajectory.png').is_file()
    finally:
        plt.close('all')
