# hello-vr

Minimal OpenXR "hello world" for SteamVR, written in Rust with the
[`openxr`](https://crates.io/crates/openxr) and [`ash`](https://crates.io/crates/ash) crates.
It renders a rotating cube above a floor quad with "Hello, World!" floating above it,
both eyes in one pass via Vulkan multiview. A red button floats in front of the
cube: point at it with either controller's laser and pull the trigger to swap
the room for the first stereoscopic photo. While a photo is shown, any trigger
click (no aiming needed) shows the next one, and after the last one you're back
in the room. The controller buzzes on every click. Any `<name>.left.jpg` with a matching `<name>.right.jpg` in
`assets/images/` is a stereo pair; pairs are shown in name order. Each eye sees
its own image, spanning the camera's field of view. A photo appears straight
ahead and level, then stays put as you turn your head; it has no parallax when
you move, since its depth comes only from the two images. The room, button and
controllers are hidden. The field of
view comes from the EXIF focal length: the 35 mm equivalent if present,
otherwise the real focal length with an assumed 1/2" phone sensor
(`ASSUMED_SENSOR_DIAGONAL_MM` in `src/photo.rs`). Bindings are suggested for
the Khronos simple controller, Valve Index, HTC Vive and Oculus Touch profiles. The text comes from a 5x7 bitmap font baked
into the fragment shader, so the stereo photo is the only texture.

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
- `src/photo.rs` - finds stereo pairs, decodes them (honouring EXIF rotation), derives their field of view from EXIF, and uploads each as a two-layer texture
- `src/shaders/scene.wgsl` - vertex and fragment shader; geometry is generated from the vertex index and text from an embedded bitmap font

## Tests

```sh
cargo test
```

Checks that the shader compiles, the push constant block fits Vulkan's 128 byte
minimum, and the projection matrix maps near/far to Vulkan depth range.
