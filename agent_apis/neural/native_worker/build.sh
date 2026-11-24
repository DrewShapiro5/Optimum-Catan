#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release --manifest-path "$HERE/Cargo.toml" "$@"
