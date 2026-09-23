#!/usr/bin/env bash
# Rebuilds on every save. Takes the same arguments as build.sh.
#
#   ./watch.sh          watch every assignment
#   ./watch.sh 2        watch only assignment 2
#
# Leave a viewer that reloads on its own open on build/pdf/assignment-<n>.pdf
# (zathura does) and the PDF follows the Markdown a couple of seconds behind.
# Press space in this terminal to force a rebuild, q or Ctrl-C to stop.

set -euo pipefail

cd "$(dirname "$0")"

command -v entr >/dev/null || {
  echo "entr is not installed" >&2
  exit 1
}

# Everything build.sh reads. find picks up a new content/<n>/ without editing
# this list, and avoids the shell's ls alias, which entr can't parse.
sources() {
  find content -name index.md
  printf '%s\n' build.sh templates/page.html static/style.css
}

# entr reads the keyboard from the terminal, so it refuses to start without one.
# -n turns that off, for when this runs from a script or a service.
opts=(-d)
[[ -t 0 ]] || opts+=(-n)

# entr -d exits when a watched file is replaced, which is what an editor saving
# through a temp file and a rename looks like from inotify's side. The loop
# restarts it on the new file, and the sleep in the condition means Ctrl-C lands
# there and ends the loop rather than just the entr inside it.
while sleep 0.5; do
  status=0
  sources | entr "${opts[@]}" ./build.sh "$@" || status=$?
  [[ $status -eq 2 ]] || exit "$status"
done
