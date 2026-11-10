# hello-vr

Minimal OpenXR "hello world" for SteamVR, written in Rust with the
[`openxr`](https://crates.io/crates/openxr) and [`ash`](https://crates.io/crates/ash) crates.
It renders a rotating cube above a floor quad with "Hello, World!" floating above it,
both eyes in one pass via Vulkan multiview. The text comes from a 5x7 bitmap font baked
into the fragment shader, so there are no textures, fonts or descriptor sets.

## Requirements

- Rust (pinned in `mise.toml`, run `mise install` once)
- A Vulkan 1.1 capable GPU and driver
- Steam with SteamVR installed and set as the active OpenXR runtime
  (`~/.config/openxr/1/active_runtime.json` should point at SteamVR)
- A headset connected and visible to SteamVR

No shader compiler is needed: `src/shaders/scene.wgsl` is compiled to SPIR-V at
startup by naga.

## Running

```sh
cargo run --release
```

The OpenXR loader is found in this order: `$OPENXR_LOADER`, the system library
path, then the copy bundled with SteamVR. Press Ctrl-C to exit; the app asks the
runtime to end the session and waits for it before shutting down.

If SteamVR is not running it is started automatically. If no headset is
connected, instance creation fails and `~/.steam/steam/logs/vrserver.txt` will
say `Refusing connect ... VRInitError_Driver_WirelessHmdNotConnected`.

To try it without hardware, enable SteamVR's null driver in
`~/.steam/steam/config/steamvr.vrsettings`:

```json
"driver_null": { "enable": true },
"steamvr": { "forcedDriver": "null", "activateMultipleDrivers": true }
```

## Layout

- `src/main.rs` - OpenXR instance/session setup, Vulkan device and pipeline, frame loop
- `src/shaders/scene.wgsl` - vertex and fragment shader; geometry is generated from the vertex index and text from an embedded bitmap font

## Tests

```sh
cargo test
```

Checks that the shader compiles, the push constant block fits Vulkan's 128 byte
minimum, and the projection matrix maps near/far to Vulkan depth range.
