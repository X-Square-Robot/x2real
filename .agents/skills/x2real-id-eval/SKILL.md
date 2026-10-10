---
name: x2real-id-eval
description: Discover, validate, compose, launch, and diagnose X2Real ID evaluations from the current ManaEnv checkout. Use when asked to run or provide a command for an X2Real ID case with any available board, robot, exact environment, policy profile, endpoint, episode budget, or Kit/headless visualization. Excludes OOD, preview, teleoperation, data generation, and policy-service deployment.
---

# X2Real ID Evaluation

Launch the requested X2Real ID evaluation from repository-owned catalogs. Treat the
board, case, policy, endpoint, budget, and visualization mode as selections, not
skill defaults.

## Read live sources

Resolve the repository root first and inspect these sources rather than carrying a
cached board, robot, case, or policy list:

- `manaenv/defination/x2real/task/boards.yaml` — available boards and budgets.
- The `leaderboard_csv` selected by that board — formal ID cases.
- `manaenv/scripts/x2real_eval.py --list` — authoritative resolved case names.
- `manaenv/defination/policy/*.yaml` — client wire and model profiles.
- `manaenv/defination/embodiments/policy_space/*.yaml` — robot wire contracts.
- `manaenv/defination/x2real.yaml` — benchmark output and video recording.

Use the active project interpreter when it imports the checkout correctly. Otherwise,
locate the environment prepared by this repository's installer. Do not embed a
developer-specific Conda or asset path in generated commands. Let the repository
`assets/` default apply unless `MANAENV_ASSET_ROOT` is already set or the user supplies
another local asset bundle.

## Resolve the requested run

1. Preserve every selection already supplied by the user: board/robot, task or exact
   environment, policy/model, endpoint, episode count, parallelism, seed, and
   visualization.
2. When no robot or board is selected, read `boards.yaml` and present the live board
   choices. Do not accept `x2real_eval.py`'s CLI default as a model or robot decision.
3. Resolve an exact case with the evaluator itself:

   ```bash
   <python> manaenv/scripts/x2real_eval.py --list --board <board>
   ```

   Apply `--task`, `--subcategory`, `--gripper`, or `--robot` filters when useful.
   Use `--env <exact-env>` for the final single-case command.
4. Select a policy YAML that matches the runtime robot and the deployed service.
   Inspect its implementation, action type, horizon, schema fields, image settings,
   observation-ingestion behavior, and session assumptions. Do not infer a model from
   an IP address or port number.
5. Obtain the endpoint from the user, selected deployment, or verified live service.
   If the intended model or endpoint remains ambiguous, ask for that choice instead of
   silently substituting another model.
6. A smoke run uses one episode and one environment only when the user asks for a
   smoke/process check. A formal evaluation retains the selected board's catalog
   budgets unless the user overrides them.

## Check the policy boundary

The evaluation command connects to an already-running policy service. Starting,
stopping, replacing, or exposing that service is a separate operation requiring an
explicit request.

Before a real run, verify the chosen policy profile agrees with live server metadata:

- protocol and protocol version;
- `policy_name`;
- observation, action, and embodiment schema pair;
- action dimension and horizon;
- session mode and observation-ingestion behavior;
- gripper convention when declared.

For Policy Space v2, use a normal `policy_space` profile. A profile that declares
`wire_protocol: openpi` is a direct native wire and is not a v2 substitute. A stateful
policy that ingests control-step observations needs an `exclusive` service session;
an inference-only profile may set `ingest_observation_each_step: false` when its
gateway accepts the current observation in `infer_actions`.

## Validate before launch

Run repository checks in proportion to the request:

```bash
<python> manaenv/scripts/x2real_eval.py --check
<python> manaenv/scripts/x2real_eval.py --asset-check
```

Use `--asset-check` when the asset root or asset state is uncertain. It validates the
formal reachable asset closure; it does not prove Kit, rendering, or model inference.

Build the final invocation once with `--dry-run`. Confirm it reports exactly the
requested case(s), budget, parallelism, seed, policy override, endpoint, and
visualization arguments before removing `--dry-run`.

## Compose the command

Use the evaluator entry point and pass the model profile through `--policy-config`:

```bash
<python> manaenv/scripts/x2real_eval.py \
  --board <board> \
  --env <exact-env> \
  --num-episodes <episodes> \
  --num-parallel-envs <parallel-envs> \
  --policy-config policy/<policy-file>.yaml \
  --server-address <host> \
  --server-port <port> \
  --enable_cameras \
  <visualization-args>
```

The evaluator is headless by default, so omit visualization flags for the
normal command. Choose an override only when the user requests a different mode:

- Kit window: add `--viz kit`.
- Explicit headless camera/video mode: `--headless --viz none` (optional and
  equivalent to the default).

Do not add a separate recorder flag when the generated suite already retains
`output[].record_video: true`. Use `--save-path` only when the user requests another
output directory.

For multiple sequential cases, invoke one exact `--env` per subprocess and keep a
separate log/report directory per case. Do not turn a requested sequential run into a
parallel run.

## Observe the real run

When execution is requested, continue through these observable boundaries:

1. Kit/Isaac application starts with the selected visualization mode.
2. The exact environment is built and cameras produce valid frames.
3. The policy endpoint connects and its metadata/schema negotiation succeeds.
4. At least one inference returns a finite action chunk with the declared shape.
5. The episode ends and the benchmark report is written; when recording is enabled,
   confirm the video artifact is present.

On failure, identify the boundary first: catalog/config, local assets, Kit/camera,
Policy Space transport, schema/session contract, model inference, action decoding, or
episode reporting. Preserve the exact command and first causal error.

## Report outcomes precisely

Keep these outcomes separate:

- **Static selection:** catalog/config/asset checks passed.
- **Execution coverage:** the environment completed an end-to-end episode without a
  runner error.
- **Semantic success:** the episode record itself reports `success=true`.

`evaluation complete: N passed` can mean runner completion; it is not sufficient
evidence of task success. Inspect the benchmark report or `MANA_EVAL_EPISODE` record
for semantic status. Static checks and `--dry-run` are not end-to-end validation.

The task is complete when the requested command is provided, or—when execution was
requested—the selected episode reaches its report/video boundary and the final answer
states execution coverage, semantic success, and any remaining external-service risk.
