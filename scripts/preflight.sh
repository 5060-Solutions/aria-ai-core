#!/usr/bin/env bash
# Run what CI runs, before pushing.
#
# Exists because a lint reached CI that a local check could not have caught:
# clippy was run under a feature set that did not compile the changed file, so
# it passed locally and failed in the --all-features job. The combinations
# below mirror .github/workflows/ci.yml — when that file changes, change this.
set -euo pipefail

cd "$(dirname "$0")/.."

for features in \
    "" \
    "--no-default-features" \
    "--no-default-features --features net" \
    "--no-default-features --features insights" \
    "--features uniffi"
do
    echo "== clippy ${features:-(default)} =="
    # shellcheck disable=SC2086
    cargo clippy --all-targets $features -- -D warnings
    echo "== test ${features:-(default)} =="
    # shellcheck disable=SC2086
    cargo test $features
done

# The inference job builds whisper.cpp and llama.cpp, so it is slow and last.
# SKIP_INFERENCE=1 when the change provably cannot reach those backends.
if [ "${SKIP_INFERENCE:-0}" != "1" ]; then
    echo "== clippy --all-features (inference backends) =="
    CMAKE_POLICY_VERSION_MINIMUM=3.5 cargo clippy --all-targets --all-features -- -D warnings
    echo "== test --all-features =="
    CMAKE_POLICY_VERSION_MINIMUM=3.5 cargo test --all-features
fi

echo "preflight passed"
