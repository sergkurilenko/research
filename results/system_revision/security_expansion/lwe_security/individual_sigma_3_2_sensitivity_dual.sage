# Auto-generated individually instrumented estimator input.
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
    Xe=ND.DiscreteGaussian(3.2, n=n),
    m=oo,
    tag="SEAL CKKS N8192 q[60,40,60] sigma_3_2_sensitivity",
)
print("TRANSCRIPT_SCHEMA ckks_lwe_individual_attack.raw.v1", flush=True)
print("ESTIMATOR_COMMIT 3e48ef421ec256afddb3e7d2249a77eab6e9ba12", flush=True)
print("MODEL_BEGIN sigma_3_2_sensitivity", flush=True)
print("PARAMETERS", params, flush=True)
print("ATTACK_BEGIN dual", flush=True)
started = time.monotonic()
try:
    result = LWE.dual(params, red_cost_model=RC.ADPS16)
    print("ATTACK dual ::", result, flush=True)
    print("ATTACK_STATUS completed", flush=True)
except Exception as error:
    print("ATTACK_STATUS failed", type(error).__name__, repr(error), flush=True)
    raise
finally:
    print("ATTACK_ELAPSED_SECONDS", time.monotonic() - started, flush=True)
print("MODEL_END sigma_3_2_sensitivity", flush=True)
