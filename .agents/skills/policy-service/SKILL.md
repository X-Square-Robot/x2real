---
name: policy-service
description: Discover, prepare, launch, inspect, replace, stop, or diagnose a Policy Space v2 policy service from its live checkout and policy-owned deployment files. Use when asked to run a model adapter or gateway, select a v2 deployment, verify protocol metadata and episode lifecycle, investigate service connectivity or session-mode failures, or safely replace an existing policy service. Excludes ManaEnv installation and X2Real evaluator execution.
---

# Manage Policy Space v2 Services

Treat the selected policy's deployment files as the authority. Do not bake a model,
robot, GPU, machine, checkpoint, host, or port into the workflow.

## Discover the live deployment

1. Locate the Policy Space checkout used by the requested installation. It is commonly
   an installer-managed checkout under `3rd/policy_space`, but may live under a
   configured dependency root or a policy-specific model environment.
2. Inspect the checkout branch, revision, worktree state, package provenance, and
   available `policies/*/deploy.yaml` files. Preserve dirty files and nested source
   checkouts.
3. Resolve the policy from the user's request or the ManaEnv policy configuration.
   When the choice is still ambiguous, list compatible deployments and ask for the
   one decision that changes the service. Never silently select a policy.
4. Read the selected policy's `deploy.yaml`, `README.md`, `requirements.txt` or other
   environment manifest, and model adapter. Also inspect the live CLI help and
   `docs/deployment.md`:

   ```bash
   <policy-space-bin> --help
   <policy-space-bin> check --help
   <policy-space-bin> serve --help
   ```

Use `$x2real-install` for the ManaEnv client stack. Policy-specific frameworks,
checkpoints, and service processes belong to this service workflow instead.

## Identify the topology

Classify the selected deployment before installing or starting anything:

- **In-process adapter:** `policy-space serve` loads the model and checkpoint directly.
- **Native backend plus v2 gateway:** start the policy-owned native inference service
  first, validate it through its native protocol, and then start the Policy Space v2
  adapter on the evaluator-facing endpoint.
- **Template or dummy adapter:** suitable only for protocol smoke tests. Report it as
  a smoke service, never as the requested trained model.

Follow the policy README for its native launch command and required environment. Do
not infer a native backend command from a different policy.

## Respect process and mutation boundaries

Read-only source, process, socket, GPU, and log inspection is safe. Starting a service
is authorized when the user explicitly asks to start that service. Stopping,
restarting, replacing, rebinding, or exposing a process requires an explicit request
covering the exact target.

Before changing a live service:

1. Resolve its PID, owner, parent, command line, executable, working directory,
   environment where available, listener, and log location.
2. Refuse to terminate another user's process or an ambiguous listener.
3. Record the previous launch command and configuration needed to restore it.
4. Use graceful termination and wait for exit. Do not delete checkpoints, caches,
   logs, environments, or source changes.

## Run preflight checks

Verify the selected policy environment separately from the ManaEnv environment:

- install or confirm the policy-owned requirements in the model environment;
- resolve required model source roots, checkpoints, credentials, and environment
  variables from the policy README or deployment file;
- confirm the requested GPU is visible and has enough free capacity;
- validate native-backend reachability when the deployment is a gateway;
- check both IPv4 and IPv6 listeners and the exact requested bind address;
- distinguish a listening port from established, stale, or `CLOSE_WAIT` connections
  that happen to use the same numeric source port;
- determine whether the evaluator reaches a loopback address, private interface,
  tunnel, proxy, or public mapping.

Bind to `127.0.0.1` for local use or tunneling unless the user requests remote access.
Binding to `0.0.0.0` does not prove a public endpoint works. For untrusted networks,
require an external WSS, authentication, access-control, and rate-limit layer rather
than presenting the raw WebSocket service as secure.

## Validate session semantics

Read `server.session_mode` from the selected `deploy.yaml`:

- Use `exclusive` for policies that retain observation history, KV cache, recurrent
  state, or other client-specific state in the model process.
- Use `multiplexed_serial` only when each inference contains the complete required
  observation and the model has no unisolated client-specific temporal state.
- `ingest_observation` is not available to a client in `multiplexed_serial` mode.
  Match the ManaEnv client's `ingest_observation_each_step` behavior to the declared
  server mode and model contract.

When uncertain, keep `exclusive`. Do not change the mode merely to suppress a client
error; prove that the adapter's state is safely isolated first.

## Check and launch

Install the Policy Space package into the selected policy environment only when it is
missing or its provenance is wrong:

```bash
<policy-python> -m pip install -e <policy-space-checkout>
```

Run the deployment check before serving:

```bash
<policy-space-bin> check <policy-deploy.yaml> <schema-selection-if-required>
```

`policy-space check` is not configuration-only. It loads the adapter and executes
`initialize_episode`, `ingest_observation`, model inference, action normalization,
and `finalize_episode`; it may load a large checkpoint and consume GPU memory. A
gateway deployment therefore needs its native backend running before this check.

Launch with the selected deployment and explicit endpoint:

```bash
<policy-space-bin> serve <policy-deploy.yaml> --host <bind-host> --port <bind-port>
```

Use the repository's existing service manager when one is configured. Otherwise keep
the foreground process attached while diagnosing. Create an explicit user-owned log
and PID path for a background service only when persistent background execution was
requested.

For a replacement, start and validate prerequisite native services first, confirm the
old evaluator-facing PID again, terminate it gracefully, then bind the new service.
If the replacement fails after the old service was stopped, restore the recorded old
command when doing so stays within the user's requested replacement scope.

## Verify the requested endpoint

Treat these as separate gates and report each one:

1. The expected process remains alive.
2. The exact address and port are listening.
3. A client reaches the intended endpoint, not merely a local substitute.
4. Metadata identifies Policy Space protocol v2 and the selected policy, schema pair,
   embodiment constraints, action dimension or horizon, gripper convention, and
   session mode expected by the evaluator.
5. One full episode lifecycle succeeds over the service endpoint:
   `initialize_episode`, the mode-appropriate observation path, `infer_actions`, and
   `finalize_episode`.
6. The requested real model returns finite actions with the declared shape. A
   template, zero-action adapter, metadata handshake, or open socket does not satisfy
   real-model inference.

Use `$x2real-id-eval` only after the service endpoint is ready when the user also asks
to compose or run a ManaEnv X2Real ID evaluation.

## Diagnose at the first failing boundary

Classify the first causal failure before changing configuration:

- checkout, policy files, environment, dependency, checkpoint, credential, or GPU;
- native backend startup, gateway translation, or native protocol;
- address binding, port ownership, routing, NAT, firewall, TLS, or public mapping;
- protocol version, metadata, schema pair, embodiment, action shape, or convention;
- session mode, episode ownership, or observation-ingestion behavior;
- model inference, finite-value validation, action conversion, or latency timeout.

Do not replace a real adapter with a template, weaken schema checks, switch session
mode, or extend timeouts without evidence that it addresses the first causal failure.

## Completion report

Report the selected policy and topology, source revision and environment, native and
gateway PIDs/endpoints/logs where applicable, metadata/lifecycle/inference results,
local versus evaluator-visible reachability, whether the model was real or a smoke
adapter, and every skipped gate. Call the service ready only when the endpoint
requested by the user completes the required v2 lifecycle and, for a real-model
request, returns a valid real inference result.
