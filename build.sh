#!/usr/bin/env bash
# Builds every assignment in content/ into the site in docs/ and a PDF in build/pdf/.
#
#   ./build.sh          build everything
#   ./build.sh 2        build only assignment 2
#
# The PDF is headless Chromium printing the same HTML the site serves, so the two
# can't drift. Anything that should differ lives in the @media print block in
# static/style.css.

set -euo pipefail

cd "$(dirname "$0")"

REPO_URL="https://github.com/LukeZhang0826/SYDE572"
SITE_URL="https://lukeypookster.com/SYDE572"
TOTAL_ASSIGNMENTS=5

OUT=docs
PDF=build/pdf
TMP=build/tmp

for tool in pandoc chromium; do
  command -v "$tool" >/dev/null || {
    echo "$tool is not installed" >&2
    exit 1
  }
done

# Assignment directories that actually exist, in numeric order
mapfile -t available < <(find content -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n)

if [[ $# -gt 0 ]]; then
  targets=("$@")
else
  targets=("${available[@]}")
fi

rm -rf "$TMP"
mkdir -p "$OUT" "$PDF" "$TMP"

# Pages is configured to serve this folder, and .nojekyll stops it rewriting anything
touch "$OUT/.nojekyll"
cp static/style.css "$OUT/style.css"

# The navbar is generated per page so that adding content/<n>/ is the only step
# needed to link a new assignment from every existing one.
write_nav() {
  local current=$1 file=$2 n

  {
    echo '<nav class="nav">'
    echo '  <div class="nav-inner">'
    echo '    <a class="nav-brand" href="../">SYDE 572</a>'
    echo '    <ul class="nav-links">'

    for ((n = 1; n <= TOTAL_ASSIGNMENTS; n++)); do
      if [[ " ${available[*]} " != *" $n "* ]]; then
        printf '      <li><span aria-disabled="true">Assignment %s</span></li>\n' "$n"
      elif [[ $n == "$current" ]]; then
        printf '      <li><a href="../%s/" aria-current="page">Assignment %s</a></li>\n' "$n" "$n"
      else
        printf '      <li><a href="../%s/">Assignment %s</a></li>\n' "$n" "$n"
      fi
    done

    printf '      <li class="nav-external"><a href="%s">GitHub ↗</a></li>\n' "$REPO_URL"
    echo '    </ul>'
    echo '  </div>'
    echo '</nav>'
  } >"$file"
}

for n in "${targets[@]}"; do
  src="content/$n/index.md"

  if [[ ! -f $src ]]; then
    echo "skipping assignment $n, no $src" >&2
    continue
  fi

  echo "building assignment $n"

  mkdir -p "$OUT/$n"
  write_nav "$n" "$TMP/nav-$n.html"

  pandoc "$src" \
    --from markdown \
    --to html5 \
    --standalone \
    --template templates/page.html \
    --include-before-body "$TMP/nav-$n.html" \
    --katex \
    --toc \
    --toc-depth=2 \
    --syntax-highlighting tango \
    --variable root=../ \
    --variable pageurl="$SITE_URL/$n/" \
    --variable pagelabel="${SITE_URL#https://}/$n/" \
    --variable repourl="$REPO_URL" \
    --variable repolabel="${REPO_URL#https://}" \
    --output "$OUT/$n/index.html"

  if [[ -d content/$n/media ]]; then
    # Copy the contents rather than the folder, so a rerun doesn't nest media/media
    rm -rf "${OUT:?}/$n/media"
    mkdir -p "$OUT/$n/media"
    cp -r "content/$n/media/." "$OUT/$n/media/"
  fi

  # --virtual-time-budget waits for KaTeX to finish typesetting before printing
  chromium \
    --headless \
    --disable-gpu \
    --no-pdf-header-footer \
    --virtual-time-budget=15000 \
    --print-to-pdf="$PDF/assignment-$n.pdf" \
    "file://$PWD/$OUT/$n/index.html" 2>/dev/null

  echo "  $OUT/$n/index.html"
  echo "  $PDF/assignment-$n.pdf"
done

# No home page yet, so the site root sends visitors to the first assignment
first=${available[0]}
cat >"$OUT/index.html" <<HTML
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>SYDE 572 | Luke Zhang</title>
    <meta http-equiv="refresh" content="0; url=./$first/" />
    <link rel="canonical" href="./$first/" />
  </head>
  <body>
    <p><a href="./$first/">Go to Assignment $first</a></p>
  </body>
</html>
HTML

rm -rf "$TMP"
