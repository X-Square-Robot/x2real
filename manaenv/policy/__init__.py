"""Policy module for manaenv.

Unified three-component policy shell (policy-schema-contract.md)::

    from manaenv.policy.mana_policy import ManaPolicyCFG, ManaPolicy, create_policy
    from manaenv.policy.policy_implementation.wallx import (
        WallxImpl,
        WallxImplCFG,
    )
    from manaenv.policy.policy_remote.websocket import WebSocketRemote, WebSocketRemoteCFG
    from manaenv.policy.policy_embodiment import PolicyEmbodimentCFG

Discovery providers are registered by the startup discovery bootstrap.
Use explicit submodule imports per the hard rule against re-export
aggregation in ``__init__.py``.
"""
