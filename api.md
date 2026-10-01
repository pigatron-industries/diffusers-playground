# Generic Workflow API

A REST API for running **any** registered workflow by name. Instead of hard-coding
how a specific workflow's inputs are populated, you pass the workflow name plus a
flat map of input values and the server applies them generically.

The server runs on port `8070` (see `app_nicegui.py`). Base URL below assumes
`http://localhost:8070`.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/generic/workflows` | List all registered workflows |
| `GET`  | `/api/generic/workflows/{workflow}/params` | Describe a workflow's input keys, node types, and sample values |
| `POST` | `/api/generic/run` | Run a workflow synchronously (blocks until finished) |
| `POST` | `/api/generic/async/run` | Run a workflow in the background (returns immediately) |
| `GET`  | `/api/async` | Poll the status/result of the background job |

---

## Request body (`POST /api/generic/run` and `/api/generic/async/run`)

```json
{
  "workflow": "<WorkflowName>",
  "batch_size": 1,
  "params": {
    "<input_key>": <value>,
    "<input_key>": <value>
  }
}
```

- **`workflow`** (string, required) — the workflow class name, e.g.
  `ImageDiffusionConditioningWorkflow` or `MinimaxH3VideoWorkflow`.
- **`batch_size`** (int, optional, default `1`) — number of outputs to produce.
- **`params`** (object, optional) — a flat map of input values. The keys are the
  workflow's **input param paths** (no `.value` suffix). Omit a key to use that
  input's default.

### How input keys work

Each key is the dotted path to a user-input node in the workflow, e.g. `prompt`,
`size`, `first_image`. Some workflows wrap an input inside a processor node, in which
case the path is nested — e.g. `prompt.prompt` in `ImageDiffusionConditioningWorkflow`.

**Discover the exact keys for any workflow** rather than guessing:

```bash
curl http://localhost:8070/api/generic/workflows/ImageDiffusionConditioningWorkflow/params
```

returns something like:

```json
{
  "models":           { "node_type": "DiffusionModelUserInputNode", "value": ["sd_1_5", [["realisticVisionV50_v51", 1.0]]] },
  "loras":            { "node_type": "LORAModelUserInputNode",     "value": [] },
  "size":             { "node_type": "SizeUserInputNode",          "value": [512, 512] },
  "prompt.prompt":    { "node_type": "TextAreaInputNode",          "value": "" },
  "negprompt":        { "node_type": "TextAreaInputNode",          "value": "" },
  "steps":            { "node_type": "IntUserInputNode",           "value": 40 },
  "cfgscale":         { "node_type": "FloatUserInputNode",         "value": 7.0 },
  "seed":             { "node_type": "SeedUserInputNode",          "value": null },
  "scheduler":        { "node_type": "ListSelectUserInputNode",    "value": "DPMSolverMultistepScheduler" },
  "sigmas":           { "node_type": "DictSelectUserInputNode",    "value": "None" },
  "clipskip":         { "node_type": "IntUserInputNode",           "value": null },
  "conditioning_inputs": { "node_type": "ListUserInputNode",       "value": [] }
}
```

### Value formats

Most inputs take their raw value directly (string, number, list). Two special cases:

- **Image inputs** (`first_image`, `last_image`, etc.): pass a **base64-encoded image**
  (optionally a `data:` URL) or a **local file path**, or `null` to leave it empty.
  The server decodes it and feeds it to the workflow.
- **Model / LoRA inputs**: structured lists (see the per-workflow tables below).

---

## Response format

### Synchronous (`/api/generic/run`)

Blocks until the workflow finishes, then returns the result:

```json
{
  "status": "finished",
  "action": "generic",
  "workflow": "ImageDiffusionConditioningWorkflow",
  "outputs": [
    { "type": "Image", "image": "<base64-encoded PNG>" }
  ],
  "applied_params": ["models", "size", "prompt.prompt", "negprompt", "steps", "cfgscale", "seed"],
  "unmatched_params": []
}
```

- **`status`** — `finished` or `error`.
- **`outputs`** — one entry per batch item. `type` is `Image`, `Video`, `Audio`, or `str`.
  - `Image` → `image` is a base64 string.
  - `Video` / `Audio` → `file` is the server path to the saved file.
  - `str` → `value` is the text.
- **`applied_params`** — the input keys that were actually set.
- **`unmatched_params`** — keys you sent that did not correspond to any input node
  (useful for catching typos).
- On error, `status` is `error` and an `error` field contains the message.

### Asynchronous (`/api/generic/async/run`) + poll (`/api/async`)

`/api/generic/async/run` returns immediately:

```json
{ "status": "running", "action": "generic" }
```

Poll `GET /api/async` until `status` is `finished` or `error`. When finished, the
response has the same shape as the synchronous response above.

> Note: there is a single shared job slot. Starting a second async job while one is
> running returns the in-progress job's status rather than queueing a new one.

---

## Example — `ImageDiffusionConditioningWorkflow` (text-to-image)

This workflow produces an image from a prompt. Key inputs and their value formats:

| Key | Type | Value format |
|-----|------|--------------|
| `models` | base + models | `[base, [[name, weight], ...]]` |
| `loras` | lora list | `[[name, weight], ...]` (empty `[]` for none) |
| `size` | dimensions | `[width, height]` |
| `prompt.prompt` | string | the text prompt (note the nested `prompt.prompt` path) |
| `negprompt` | string | the negative prompt |
| `steps` | int | sampling steps |
| `cfgscale` | float | guidance scale |
| `seed` | int / null | seed, or `null` for random |
| `scheduler` | string | e.g. `DPMSolverMultistepScheduler` |
| `sigmas` | string | `None` or an AYS schedule name (sets `steps` when a schedule is chosen) |
| `clipskip` | int / null | CLIP skip, or `null` |
| `conditioning_inputs` | list | advanced conditioning (see discovery endpoint) |

### Request

```bash
curl -s -X POST http://localhost:8070/api/generic/run \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "ImageDiffusionConditioningWorkflow",
    "batch_size": 1,
    "params": {
      "models": ["sd_1_5", [["realisticVisionV50_v51", 1.0]]],
      "loras": [],
      "size": [768, 512],
      "prompt.prompt": "a serene mountain lake at sunrise, ultra detailed, 4k",
      "negprompt": "blurry, low quality, watermark, text",
      "steps": 30,
      "cfgscale": 7.0,
      "seed": 42,
      "scheduler": "DPMSolverMultistepScheduler",
      "sigmas": "None",
      "clipskip": 1
    }
  }'
```

### Response

```json
{
  "status": "finished",
  "action": "generic",
  "workflow": "ImageDiffusionConditioningWorkflow",
  "outputs": [
    { "type": "Image", "image": "iVBORw0KGgoAAAANSUhEUgAAA..." }
  ],
  "applied_params": ["models", "loras", "size", "prompt.prompt", "negprompt", "steps", "cfgscale", "seed", "scheduler", "sigmas", "clipskip"],
  "unmatched_params": []
}
```

> Tip: `models[0]` is the base model and `models[1]` is a list of
> `[model_name, weight]` pairs. Use `loras: []` when you have no LoRAs. The exact
> `base` / model names depend on which models you have loaded — see the discovery
> endpoint above.

---

## Example — `MinimaxH3VideoWorkflow` (text-to-video / image-to-video)

This workflow produces a video from a prompt, optionally conditioned on a first and/or
last frame. Key inputs and their value formats:

| Key | Type | Value format |
|-----|------|--------------|
| `prompt` | string | the text prompt |
| `first_image` | image / null | base64 image, `data:` URL, or local file path; `null` for none |
| `last_image` | image / null | same as above |
| `resolution` | dimensions | `[width, height]` |
| `duration` | float | video length in seconds |
| `steps` | int | sampling steps |
| `seed` | int / null | seed, or `null` for random |

### Text-to-video request

```bash
curl -s -X POST http://localhost:8070/api/generic/run \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "MinimaxH3VideoWorkflow",
    "batch_size": 1,
    "params": {
      "prompt": "a cat walking through a misty forest, cinematic, slow motion",
      "resolution": [768, 448],
      "duration": 5.0,
      "steps": 9,
      "seed": 42
    }
  }'
```

### Image-to-video request

Provide `first_image` (and optionally `last_image`) as a base64-encoded image.
You can also pass a local file path instead of base64.

```bash
curl -s -X POST http://localhost:8070/api/generic/run \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "MinimaxH3VideoWorkflow",
    "batch_size": 1,
    "params": {
      "prompt": "a cat walking through a misty forest",
      "first_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAA...",
      "last_image": null,
      "resolution": [768, 448],
      "duration": 5.0,
      "steps": 9,
      "seed": 42
    }
  }'
```

### Response

```json
{
  "status": "finished",
  "action": "generic",
  "workflow": "MinimaxH3VideoWorkflow",
  "outputs": [
    { "type": "Video", "file": "/Users/rob/diffusers-playground/output/1699000000.mp4" }
  ],
  "applied_params": ["prompt", "first_image", "last_image", "resolution", "duration", "steps", "seed"],
  "unmatched_params": []
}
```

> Note: for a video, the output `file` is a path on the server. Fetch it over HTTP or
> copy it locally. `first_image` / `last_image` are optional — omit them (or send
> `null`) for pure text-to-video.

