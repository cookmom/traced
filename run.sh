#!/bin/bash
# <!-- بسم الله الرحمن الرحيم -->
# Traced Pipeline — Full end-to-end
# Usage: ./run.sh <image> <name> [--version vN]
# Example: ./run.sh examples/szm-reference.jpg "Sheikh Zayed Grand Mosque" --version v11
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${1:?Usage: ./run.sh <image> <name> [--version vN]}"
NAME="${2:?Usage: ./run.sh <image> <name> [--version vN]}"
VERSION="v$(date +%Y%m%d-%H%M)"
shift 2
while [[ $# -gt 0 ]]; do
    case "$1" in --version) VERSION="$2"; shift 2;; *) echo "Unknown: $1"; exit 1;; esac
done

OUTPUT="szm-${VERSION}.html"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TRACED PIPELINE — ${VERSION}"
echo "  Image: ${IMAGE}"
echo "  Name:  ${NAME}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Ensure we're on main branch
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "▸ Switching to main branch (was on ${CURRENT_BRANCH})..."
    git stash -q 2>/dev/null || true
    git checkout main -q 2>/dev/null || git checkout -b main -q
    git stash pop -q 2>/dev/null || true
fi

echo "▸ [1/6] Preprocess → 1080×1920"
python preprocess.py --image "$IMAGE" --output preprocessed.jpg
echo ""

echo "▸ [2/6] Research"
if [ -f knowledge.json ]; then
    echo "  (cached)"
else
    python research.py --name "$NAME" --output knowledge.json
fi
echo ""

echo "▸ [3/6] Extract (SAM 2 + Depth Anything)"
python extract-sam2.py --image preprocessed.jpg --knowledge knowledge.json --output extraction.json --name "$NAME"
echo ""

echo "▸ [4/6] Optimize (gradient descent + constraints)"
python optimize.py --extraction extraction.json --image preprocessed.jpg --output optimized.json
echo ""

echo "▸ [5/6] Construct (columns, drums, finials, cornices)"
python construct.py --optimized optimized.json --extraction extraction.json --output constructed.json
echo ""

echo "▸ [6/6] Generate HTML"
python generate.py --extraction extraction.json --optimized constructed.json --knowledge knowledge.json --ref-image preprocessed.jpg --output "$OUTPUT" --name "$NAME"
echo ""

# Copy as szm.html for stable URL
cp "$OUTPUT" szm.html

# Auto-deploy
echo "▸ Deploying to GitHub Pages..."
git add "$OUTPUT" szm.html preprocessed.jpg 2>/dev/null || true
git commit -m "traced: ${NAME} ${VERSION}" 2>/dev/null || true
git push origin main 2>/dev/null && {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ✅ DEPLOYED"
    echo "  https://cookmom.github.io/traced/szm.html?v=${VERSION}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
} || {
    echo ""
    echo "⚠ Push failed. Run once: git remote set-url origin git@github.com:cookmom/traced.git"
    echo "  File saved locally: $OUTPUT"
}
