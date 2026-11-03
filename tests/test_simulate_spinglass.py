"""Independent direct-field reference checks for the Monte Carlo simulator."""
import json
from dataclasses import replace
from datetime import datetime
import numpy as np
import pytest
from simulate_spinglass import (Parameters, make_couplings, initial_spins,
    flip_probability, field_distribution, read_config, run, reserve_output)


def small(**kw):
    return Parameters(n_spins=20, t_end=2.15, record_interval=.15,
                      distribution_interval=.5, **kw)


@pytest.mark.parametrize('k', [0., .3, .6, 1.])
@pytest.mark.parametrize('temperature', [0., .01, 1.5])
def test_cached_updates_against_direct_reference(k, temperature):
    p=small(asymmetry=k, temperature=temperature)
    actual=run(p)
    streams=np.random.SeedSequence(p.seed).spawn(3)
    jr, ir, rng=[np.random.Generator(np.random.PCG64(s)) for s in streams]
    j=make_couplings(p,jr)
    s=initial_spins(p,ir)
    history=[[0.,s.mean(),-.5*np.dot(s.astype(float),j@s)/p.n_spins]]
    accepted=0
    for attempt in range(1, round(p.t_end*p.n_spins)+1):
        i=int(rng.integers(p.n_spins))
        h=float(j[i]@s)
        # Independently expressed heat-bath acceptance rule.
        aligned=int(s[i])*h
        probability=(float(aligned<0)+.5*float(aligned==0)) if temperature==0 else 1/(1+np.exp(np.clip(2*aligned/temperature,-700,700)))
        if rng.random()<probability:
            s[i]=-s[i]; accepted+=1
        if attempt % round(p.record_interval*p.n_spins)==0 or attempt==round(p.t_end*p.n_spins):
            history.append([attempt/p.n_spins,s.mean(),-.5*np.dot(s.astype(float),j@s)/p.n_spins])
    np.testing.assert_array_equal(actual['spins'],s)
    np.testing.assert_allclose(actual['fields'],j@s,atol=2e-14)
    np.testing.assert_allclose(actual['trajectory'][:,:3],history,atol=2e-14)
    assert actual['accepted_updates']==accepted
    tr=actual['trajectory']
    np.testing.assert_allclose(tr[1:,3:], np.diff(tr[:,1:3],axis=0)/np.diff(tr[:,0,None],axis=0))
    assert np.isnan(tr[0,3:]).all()


def test_sampling_does_not_change_path():
    p=small()
    a=run(p); b=run(replace(p,record_interval=.25,distribution_interval=.1))
    np.testing.assert_array_equal(a['spins'],b['spins'])
    np.testing.assert_array_equal(a['fields'],b['fields'])


def test_coupling_symmetry_and_statistics():
    p=Parameters(n_spins=800)
    j=make_couplings(p,np.random.default_rng(5))
    np.testing.assert_array_equal(j,j.T)
    assert not np.diag(j).any()
    for k in (0.,.5,1.):
        j=make_couplings(replace(p,asymmetry=k),np.random.default_rng(5))
        x,y=j[np.triu_indices(p.n_spins,1)],j.T[np.triu_indices(p.n_spins,1)]
        assert np.var(x)==pytest.approx(1/p.n_spins,rel=.02)
        assert np.corrcoef(x,y)[0,1]==pytest.approx((1-k*k)/(1+k*k),abs=.015)


def test_initial_population_and_zero_field():
    p=Parameters(n_spins=5,m0=.3,record_interval=.2)
    s=initial_spins(p,np.random.default_rng(2))
    assert (s==-1).sum()==2
    assert flip_probability(0.,0.)==.5
    assert flip_probability(-1.,0.)==1
    assert flip_probability(1.,0.)==0
    a=run(small(j1=0.,j0=0.,temperature=0.))
    assert 0<a['accepted_updates']<a['attempted_updates']


def test_histogram_mass_and_empty_population():
    spins=np.array([1,1,1,-1,-1])
    h=np.array([-2.,0.,1.,2.,3.])
    edges=np.array([-1.,0.,1.,2.])
    rho,outside=field_distribution(spins,h,edges)
    assert (rho*np.diff(edges)).sum()+outside.sum()/5==pytest.approx(1.)
    np.testing.assert_array_equal(outside,[[1,0],[0,1]])
    rho,outside=field_distribution(np.ones(5),np.zeros(5),edges)
    assert np.isfinite(rho).all() and not rho[1].any()


def test_zero_run_and_configuration(tmp_path):
    a=run(replace(small(),t_end=0.))
    assert a['trajectory'].shape==(1,5)
    assert a['rho_history'].shape==(1,2,120)
    config=tmp_path/'p.txt'
    config.write_text('seed = 13 # comment\noutput = some path.npz\n')
    assert read_config(config).seed==13
    for text in ('seed=2.0','unknown=3','seed=1\nseed=2','temperature=nan','output='):
        config.write_text(text)
        with pytest.raises(ValueError): read_config(config)
    with pytest.raises(ValueError): Parameters(t_end=.0001)
    with pytest.raises(ValueError): Parameters(record_interval=.0001)
    dt=datetime(2026,10,19)
    p,f=reserve_output(tmp_path/'test.npz',dt); f.close()
    p2,f=reserve_output(tmp_path/'test.npz',dt); f.close()
    assert p!=p2 and p.exists()


def test_energy_uses_symmetric_part_and_no_self_interactions():
    p=small(asymmetry=1.,j0=.4)
    j=make_couplings(p,np.random.default_rng(7))
    s=initial_spins(p,np.random.default_rng(8)).astype(float)
    symmetric=(j+j.T)/2
    assert s@(j@s)==pytest.approx(s@(symmetric@s))
    site=3
    changed=s.copy(); changed[site]*=-1
    # For the symmetric contribution, this is the exact energy change per spin.
    delta=(-.5*changed@(j@changed)+.5*s@(j@s))/p.n_spins
    assert delta==pytest.approx(2*s[site]*(symmetric@s)[site]/p.n_spins)


def test_cli_archive_metadata_and_collision(tmp_path):
    from simulate_spinglass import main
    config=tmp_path/'simulation.txt'
    config.write_text(f'n_spins=20\nt_end=0.15\nrecord_interval=0.1\n'
                      f'output={tmp_path}/simulation.npz\nseed=19\n')
    main([str(config)])
    main([str(config)])
    archives=list(tmp_path.glob('simulation_*.npz'))
    assert len(archives)==2
    with np.load(archives[0],allow_pickle=False) as data:
        p=json.loads(data['parameters'].item())
        assert p['seed']==19 and p['config']==str(config)
        assert p['output_file']==str(archives[0])
        assert data['run_type'].item()=='spin_simulation'
        assert 'q' not in data['trajectory_columns']
        np.testing.assert_allclose(data['trajectory'][:,0],[0.,.1,.15])
        np.testing.assert_array_equal(data['energy'],data['trajectory'][:,2])
        assert data['attempted_updates']==3


@pytest.mark.parametrize('k', [0., .6, 1.])
def test_fixed_coupling_ensemble_and_single_run_compatibility(k, monkeypatch):
    import simulate_spinglass as simulator
    p=small(asymmetry=k, n_runs=3, seed=17)
    original=make_couplings
    calls=[]
    def counted(p, rng):
        matrix=original(p,rng)
        calls.append(matrix)
        return matrix
    monkeypatch.setattr(simulator,'make_couplings',counted)
    ensemble=simulator.run_ensemble(p)
    assert len(calls)==1
    shared=calls[0]
    np.testing.assert_array_equal(ensemble['run_seeds'],['17','18','19'])
    for i in range(3):
        np.testing.assert_allclose(ensemble['run_fields'][i],shared@ensemble['run_spins'][i],atol=1e-14)
        ip=replace(p,seed=p.seed+i,n_runs=1)
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(ip.seed).spawn(3)[1]))
        np.testing.assert_array_equal(ensemble['run_initial_spins'][i],initial_spins(ip,rng))
        reference=run(ip,_couplings=shared)
        np.testing.assert_allclose(ensemble['trajectories'][i],reference['trajectory'],equal_nan=True)
    assert not np.array_equal(ensemble['run_initial_spins'][0],ensemble['run_initial_spins'][1])
    old=run(replace(p,n_runs=1))
    np.testing.assert_allclose(ensemble['trajectories'][0],old['trajectory'],equal_nan=True)
    singleton=simulator.run_ensemble(replace(p,n_runs=1))
    assert set(singleton)==set(old)
    np.testing.assert_array_equal(singleton['trajectory'],old['trajectory'])
    again=simulator.run_ensemble(p)
    np.testing.assert_array_equal(ensemble['trajectories'],again['trajectories'])


@pytest.mark.parametrize('value',[0,-1,1.5])
def test_invalid_run_count(value):
    with pytest.raises(ValueError): Parameters(n_runs=value)


def test_ensemble_archive_and_zero_duration(tmp_path):
    from simulate_spinglass import main
    config=tmp_path/'ensemble.txt'
    config.write_text(f'n_spins=20\nt_end=0\nrecord_interval=0.1\nn_runs=3\n'
                      f'output={tmp_path}/ensemble.npz\nseed=10\n')
    main([str(config)])
    archive=next(tmp_path.glob('ensemble_*.npz'))
    with np.load(archive,allow_pickle=False) as data:
        metadata=json.loads(data['parameters'].item())
        assert metadata['n_runs']==3
        info=json.loads(data['ensemble'].item())
        assert info['coupling_seed']==10 and info['run_seeds']==[10,11,12]
        assert data['trajectories'].shape==(3,1,5)
        np.testing.assert_array_equal(data['trajectory'],data['trajectories'][0])
        assert data['run_rho_history'].shape==(3,1,2,120)
        assert data['run_simulation'].shape==(3,)
        np.testing.assert_array_equal(data['run_attempted_updates'],[0,0,0])


def test_combined_style_simulation_names(tmp_path):
    p = tmp_path/'sim.txt'
    p.write_text('T=.5\nsimulation_cells=120\nn_runs=20\noutput_dir=Output\noutput_prefix=simulation\n')
    settings = read_config(p)
    assert settings.temperature == .5 and settings.cells == 120 and settings.n_runs == 20
    assert settings.output == 'Output/simulation.npz'
    for text in ['cells=24\nsimulation_cells=32', 'output=a.npz\noutput_prefix=b',
                 'output_prefix=../bad', 'T=1\ntemperature=2']:
        p.write_text(text)
        with pytest.raises(ValueError): read_config(p)
