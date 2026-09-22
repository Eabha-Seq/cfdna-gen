#!/usr/bin/env bash
# Mode B — WisecondorX predict, only if a synthetic euploid reference exists.
# Never synthesises a 50-sample panel. Never imports private packages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-${SCRIPT_DIR}/work}"
OUTDIR="${OUTDIR:-${SCRIPT_DIR}/out}"
REF=""
REF_URL=""
BAM=""
BUILD_REF=""
BINSZ=100000

usage() {
  cat <<'EOF'
Usage: ./run_wx.sh [options]

Mode B: wisecondorx convert + predict, only if a euploid reference.npz exists.

Reference lookup order (first hit wins):
  1. --ref PATH
  2. env EABHASEQ_WX_REF
  3. ./ref/synthetic_euploid_nipt.npz
  4. --ref-url URL   (downloaded to work/downloaded_ref.npz)

If none exist this script prints the newref command and exits 0.
It does not generate 50 full-depth samples.

Options:
  --ref PATH          Existing synthetic euploid .npz
  --ref-url URL       Download a .npz reference
  --bam PATH          BAM from Mode A (default: ./work/sample.bam)
  --workdir DIR       Working directory (default: ./work)
  --outdir DIR        Predict output directory (default: ./out)
  --build-ref DIR     Run newref --nipt if DIR contains >=50 euploid .npz
  --binsize N         newref binsize (default: 100000)
  -h, --help          Show this help
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

wx_bin() {
  if command -v wisecondorx >/dev/null 2>&1; then
    command -v wisecondorx
  elif command -v WisecondorX >/dev/null 2>&1; then
    command -v WisecondorX
  else
    return 1
  fi
}

print_missing_ref() {
  cat <<EOF
WisecondorX predict needs a euploid reference.npz.
The free zip is one labelled T21 file. It is not a reference panel.

EabhaSeq's published panel was 50 synthetic euploid samples (same mapper,
GRCh38, and synthetic material type as the test BAM). WisecondorX docs
require >=50 negative controls.

To build a panel from already-converted euploid npz (do not run this
unless you already have >=50 files — this recipe will not generate them):

  wisecondorx newref euploid_npz/*.npz ref/synthetic_euploid_nipt.npz --nipt --binsize 100000

Then:
  ./run_wx.sh --ref ref/synthetic_euploid_nipt.npz

Or set EABHASEQ_WX_REF / pass --ref-url.

reference missing; Mode A complete
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --ref)
      REF="$2"
      shift 2
      ;;
    --ref-url)
      REF_URL="$2"
      shift 2
      ;;
    --bam)
      BAM="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --build-ref)
      BUILD_REF="$2"
      shift 2
      ;;
    --binsize)
      BINSZ="$2"
      shift 2
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

mkdir -p "$WORKDIR" "$OUTDIR"
BAM="${BAM:-${WORKDIR}/sample.bam}"
DEFAULT_REF="${SCRIPT_DIR}/ref/synthetic_euploid_nipt.npz"

if [[ -n "$BUILD_REF" ]]; then
  WX="$(wx_bin)" || die "wisecondorx not found
  conda env create -f environment.yml
  conda activate eabhaseq-wx"
  mapfile -t NPZ < <(find "$BUILD_REF" -maxdepth 1 -type f -name '*.npz' | sort)
  if [[ ${#NPZ[@]} -lt 50 ]]; then
    die "--build-ref needs >=50 euploid .npz (found ${#NPZ[@]}). This recipe will not generate them."
  fi
  mkdir -p "${SCRIPT_DIR}/ref"
  printf 'Building NIPT reference from %s files ...\n' "${#NPZ[@]}"
  "$WX" newref "${NPZ[@]}" "$DEFAULT_REF" --nipt --binsize "$BINSZ"
  REF="$DEFAULT_REF"
fi

if [[ -z "$REF" && -n "${EABHASEQ_WX_REF:-}" ]]; then
  REF="$EABHASEQ_WX_REF"
fi
if [[ -z "$REF" && -f "$DEFAULT_REF" ]]; then
  REF="$DEFAULT_REF"
fi
if [[ -z "$REF" && -n "$REF_URL" ]]; then
  REF="${WORKDIR}/downloaded_ref.npz"
  printf 'Downloading reference %s\n' "$REF_URL"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$REF" "$REF_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$REF" "$REF_URL"
  else
    die "need curl or wget to fetch --ref-url"
  fi
fi

if [[ -z "$REF" || ! -f "$REF" ]]; then
  print_missing_ref
  exit 0
fi

WX="$(wx_bin)" || die "wisecondorx not found
  conda env create -f environment.yml
  conda activate eabhaseq-wx"

[[ -f "$BAM" ]] || die "BAM not found: $BAM
Run Mode A first:
  ./run_qc.sh /path/to/GRCh38.fa"

SAMPLE_NPZ="${WORKDIR}/sample.npz"
OUT_PREFIX="${OUTDIR}/t21"

printf 'Using reference: %s\n' "$REF"
printf 'Convert + predict (WisecondorX CLI, no private packages) ...\n'
"$WX" convert "$BAM" "${WORKDIR}/sample"
"$WX" predict "$SAMPLE_NPZ" "$REF" "$OUT_PREFIX" --bed --plot \
  || "$WX" predict "$SAMPLE_NPZ" "$REF" "$OUT_PREFIX" --bed

python3 "${SCRIPT_DIR}/parse_wx_stats.py" "$OUTDIR"
echo "Mode B complete (pipeline smoke test, not a clinical call)."
