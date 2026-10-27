#!/usr/bin/env bash
# Run on the remote login node, inside the Ubuntu 22.04 x86_64 container.
# Download and unpack pinned software-rendering libraries without root access.
set -euo pipefail
source /etc/os-release
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 || ${ID:-} != ubuntu || ${VERSION_ID:-} != 22.04 ]]; then
    echo "Run inside the remote Ubuntu 22.04 x86_64 container." >&2
    exit 1
fi
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?Usage: prepare-render-libs.sh /absolute/node-local/output-directory}
[[ "$OUT" = /* && ! -e "$OUT" ]] || { echo "Use a new absolute output directory." >&2; exit 1; }
mkdir -p "$OUT/debs/lists/partial" "$OUT/root"
cd "$OUT/debs"
apt-get -o Dir::State::lists="$PWD/lists" -o APT::Get::List-Cleanup=0 update
mapfile -t packages < "$HERE/render-packages.txt"
apt-get -o Dir::State::lists="$PWD/lists" download "${packages[@]}"
sha256sum -c "$HERE/render-packages.sha256"
for package in ./*.deb; do
    dpkg-deb -x "$package" "$OUT/root"
done
tar -C "$OUT/root" -cf "$OUT/render-libs.tar" usr
printf 'Prepared %s\n' "$OUT/render-libs.tar"
