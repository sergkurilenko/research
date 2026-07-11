# Auto-generated extended/default-cost estimator point.
# Estimator commit: 3e48ef421ec256afddb3e7d2249a77eab6e9ba12
from estimator import *
from sage.all import oo
import time

n = 8192
q = Integer(1461501441329443417981674104123436284398337261569)
params = LWE.Parameters(
    n=n,
    q=q,
    Xs=ND.Uniform(-1, 1, n=n),
    Xe=ND.CenteredBinomial(21, n=n),
    m=oo,
    tag="SEAL CKKS N8192 q[60,40,60] seal_exact_cbd21",
)
print("TRANSCRIPT_SCHEMA ckks_lwe_extended_individual.raw.v1", flush=True)
print("ESTIMATOR_COMMIT 3e48ef421ec256afddb3e7d2249a77eab6e9ba12", flush=True)
print("COST_MODEL MATZOV", flush=True)
print("SHAPE_MODEL GSA", flush=True)
print("MODEL_BEGIN seal_exact_cbd21", flush=True)
print("PARAMETERS", params, flush=True)
print("ATTACK_BEGIN usvp", flush=True)
started = time.monotonic()
try:
    result = LWE.primal_usvp(params, red_cost_model=RC.MATZOV, red_shape_model=Simulator.GSA)
    print("ATTACK usvp ::", result, flush=True)
    print("ATTACK_STATUS completed", flush=True)
except Exception as error:
    print("ATTACK_STATUS failed", type(error).__name__, repr(error), flush=True)
    raise
finally:
    print("ATTACK_ELAPSED_SECONDS", time.monotonic() - started, flush=True)
print("MODEL_END seal_exact_cbd21", flush=True)
