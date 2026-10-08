"""Future memory-consolidation boundary.

Only verified reasoning should be eligible to update retrieval metadata.  This
prototype performs no write-back, preventing derived claims from becoming
self-reinforcing evidence.
"""


def propose_reachability_update(*args, **kwargs):
    raise NotImplementedError("Reachability feedback is deferred until the core loop is validated")

