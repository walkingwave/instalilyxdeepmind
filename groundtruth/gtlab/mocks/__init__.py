"""Mock registry: every family is its grey-box ODE module run with a secret true theta."""
from gtlab.mocks.base import MockSystem, Y0_RANGE, ode_module, true_theta  # noqa: F401
from gtlab.systems import SYSTEM_IDS

MOCKS = {sid: MockSystem for sid in SYSTEM_IDS}


def get_mock(system_id, mech=("A", "B"), theta=None, noise=0.03, seed=0, **kw):
    if system_id not in MOCKS:
        raise KeyError(system_id)
    return MOCKS[system_id](system_id, mech=mech, theta=theta, noise=noise, seed=seed, **kw)
