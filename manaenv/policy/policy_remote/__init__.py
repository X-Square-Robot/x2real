"""PolicyRemote package — optional ManaPolicy component.

Each transport family (websocket / ipc / native_inproc / ...) registers a
concrete :class:`~manaenv.policy.policy_remote.base.PolicyRemoteBase` subclass.
The translator selects one from the policy yaml's reflected remote CFG section,
for example ``web_socket_remote``. When the policy yaml omits a remote section,
no remote is created.

Remote modules are pure transport — they do not understand wire-field
semantics, only marshalling request/response dicts. The remote process
typically runs the same PolicyImplementation logic; the network only relays
the encoded payload.

Use explicit submodule imports (no aggregated re-exports here per policy-schema-contract.md import rule).
"""
