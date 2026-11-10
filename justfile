# List available recipes
default:
    @just --list

# Format the code in place
format:
    cargo fmt

# Run clippy with warnings as errors and check formatting
lint:
    cargo clippy --all-targets -- -D warnings
    cargo fmt --check

# Run the unit tests
test:
    cargo test

# Build and launch the app against the active OpenXR runtime (needs SteamVR and a headset)
run:
    cargo run --release
