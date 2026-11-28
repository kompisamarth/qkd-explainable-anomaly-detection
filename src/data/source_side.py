"""Device-disjoint source datasets; counterfactual scenarios share nuisance streams."""
from concurrent.futures import ThreadPoolExecutor
from src.simulator.source_side import SourceSideSimulator


def generate_source_sessions(seed,device_ids,scenarios=range(6),profile='minimal',
                             config=None,workers=1,stress='nominal'):
    jobs=[(int(d),int(s)) for d in device_ids for s in scenarios]
    def one(job):
        d,s=job
        return SourceSideSimulator(config).simulate(seed,d,0,s,profile,stress=stress)
    if workers==1: return [one(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one,jobs))
