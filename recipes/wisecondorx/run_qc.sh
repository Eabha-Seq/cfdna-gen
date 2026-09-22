#!/usr/bin/env bash
# Mode A — alignment QC on the free labelled T21 FASTQ.
# Downloads the zip, aligns with bwa mem to operator-supplied GRCh38,
# prints a small QC table, and stops. This is not a trisomy call.
set -euo pipefail

SAMPLE_URL="${EABHASEQ_SAMPLE_URL:-https://eabhaseq.lon1.digitaloceanspaces.com/samples/t21-4m-sample.zip}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-${SCRIPT_DIR}/work}"
THREADS="${THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
SKIP_DOWNLOAD=0
SOFTCLIP=0
FASTQ_DIR=""
REF_FA=""

usage() {
  cat <<'EOF'
Usage: ./run_qc.sh /path/to/GRCh38.fa [options]

Mode A: download the free 4M PE T21 FASTQ, align with bwa mem, print QC.
Does not call trisomy. Does not download GRCh38.

Options:
  --workdir DIR       Working directory (default: ./work)
  --threads N         bwa / samtools threads (default: nproc)
  --sample-url URL    Override the FASTQ zip URL
  --fastq-dir DIR     Use already-unpacked FASTQ instead of downloading
  --skip-download     Reuse an existing zip / unpacked FASTQ in --workdir
  --softclip          Also estimate soft-clip fraction from CIGAR (slow)
  -h, --help          Show this help

If GRCh38 or bwa is missing this script exits with the install command.
It will not fetch a 3 GB genome.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    printf 'error: %s not found on PATH\n' "$cmd" >&2
    printf '%s\n' "$hint" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --sample-url)
      SAMPLE_URL="$2"
      shift 2
      ;;
    --fastq-dir)
      FASTQ_DIR="$2"
      shift 2
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --softclip)
      SOFTCLIP=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      if [[ -z "$REF_FA" ]]; then
        REF_FA="$1"
        shift
      else
        die "unexpected argument: $1"
      fi
      ;;
  esac
done

if [[ -z "$REF_FA" ]]; then
  usage >&2
  die "GRCh38 FASTA path is required"
fi

INSTALL_HINT='Install tools (do not download GRCh38 from this script):
  conda env create -f environment.yml
  conda activate eabhaseq-wx
  # then: bwa index /path/to/GRCh38.fa

Example public GRCh38 (operator download only):
  https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz'

need_cmd bwa "$INSTALL_HINT"
need_cmd samtools "$INSTALL_HINT"

if [[ ! -f "$REF_FA" ]]; then
  printf 'error: GRCh38 FASTA not found: %s\n' "$REF_FA" >&2
  printf 'This recipe does not download a 3 GB genome unasked.\n' >&2
  printf '%s\n' "$INSTALL_HINT" >&2
  exit 1
fi

if [[ ! -f "${REF_FA}.bwt" && ! -f "${REF_FA}.bwt.2bit.64" ]]; then
  printf 'error: bwa index missing next to %s\n' "$REF_FA" >&2
  printf '  bwa index %s\n' "$REF_FA" >&2
  exit 1
fi

mkdir -p "$WORKDIR"
FQ_DIR="${FASTQ_DIR:-${WORKDIR}/fastq}"
ZIP_PATH="${WORKDIR}/t21-4m-sample.zip"
BAM="${WORKDIR}/sample.bam"
STATS="${WORKDIR}/sample.stats.txt"
IDXSTATS="${WORKDIR}/sample.idxstats.txt"
QC_TABLE="${WORKDIR}/qc_table.txt"

download_zip() {
  mkdir -p "$(dirname "$ZIP_PATH")"
  if [[ -s "$ZIP_PATH" ]]; then
    printf 'Using existing zip: %s\n' "$ZIP_PATH"
    return 0
  fi
  printf 'Downloading %s\n' "$SAMPLE_URL"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$ZIP_PATH" "$SAMPLE_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$ZIP_PATH" "$SAMPLE_URL"
  else
    die "need curl or wget to download the sample zip
  conda env create -f environment.yml"
  fi
}

unpack_zip() {
  mkdir -p "$FQ_DIR"
  if find "$FQ_DIR" -type f \( -name '*.fastq.gz' -o -name '*.fq.gz' -o -name '*.fastq' -o -name '*.fq' \) | grep -q .; then
    printf 'Using FASTQ already in %s\n' "$FQ_DIR"
    return 0
  fi
  command -v unzip >/dev/null 2>&1 || die "unzip not found
  conda env create -f environment.yml"
  printf 'Unpacking %s -> %s\n' "$ZIP_PATH" "$FQ_DIR"
  unzip -o -q "$ZIP_PATH" -d "$FQ_DIR"
}

# Discover a paired-end FASTQ pair. Do not assume R1.fastq.gz.
discover_pair() {
  local dir="$1"
  mapfile -t all_fq < <(find "$dir" -type f \( \
    -name '*.fastq.gz' -o -name '*.fq.gz' -o -name '*.fastq' -o -name '*.fq' \
  \) | sort)
  if [[ ${#all_fq[@]} -eq 0 ]]; then
    die "no FASTQ files found under $dir (filenames were not assumed)"
  fi
    local r1="" r2=""
    local f base
    for f in "${all_fq[@]}"; do
      base="$(basename "$f")"
      case "$base" in
        *_R1.*|*_R1_*|*_r1.*|*_1.f*q*|*_1_*.f*q*|*.1.fastq.gz|*.1.fq.gz|*.1.fastq|*.1.fq)
          r1="$f"
          ;;
        *_R2.*|*_R2_*|*_r2.*|*_2.f*q*|*_2_*.f*q*|*.2.fastq.gz|*.2.fq.gz|*.2.fastq|*.2.fq)
          r2="$f"
          ;;
      esac
    done
  if [[ -n "$r1" && -n "$r2" ]]; then
    printf '%s\t%s\n' "$r1" "$r2"
    return 0
  fi
  if [[ ${#all_fq[@]} -eq 2 ]]; then
    printf '%s\t%s\n' "${all_fq[0]}" "${all_fq[1]}"
    return 0
  fi
  die "could not pair FASTQ under $dir; found: ${all_fq[*]}"
}

insert_peak() {
  # IS <insert size> <pairs total> ...
  awk '
    $1=="IS" {
      size=$2+0; count=$3+0
      if (count>best) { best=count; peak=size }
    }
    END { if (best>0) print peak; else print "n/a" }
  ' "$STATS"
}

gc_percent() {
  # Weighted mean of GCF / GCL histograms (GC% vs count).
  awk '
    $1=="GCF" || $1=="GCL" {
      gc=$2+0; n=$3+0
      w += gc*n; c += n
    }
    END {
      if (c>0) printf "%.2f\n", w/c
      else print "n/a"
    }
  ' "$STATS"
}

sn_field() {
  local key="$1"
  awk -F '\t' -v k="$key" '$1=="SN" && index($2, k)==1 { print $3; exit }' "$STATS"
}

softclip_frac() {
  # Optional full CIGAR scan. Off by default — 4M PE is slow.
  samtools view -F 4 "$BAM" | awk '
    {
      cigar=$6; seqlen=length($10); t+=seqlen
      while (match(cigar, /[0-9]+S/)) {
        s += substr(cigar, RSTART, RLENGTH-1)+0
        cigar = substr(cigar, RSTART+RLENGTH)
      }
    }
    END {
      if (t>0) printf "%.4f\n", s/t
      else print "n/a"
    }
  '
}

if [[ -z "$FASTQ_DIR" ]]; then
  if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
    download_zip
  elif [[ ! -s "$ZIP_PATH" ]] && ! find "$FQ_DIR" -type f \( -name '*.fastq.gz' -o -name '*.fq.gz' -o -name '*.fastq' -o -name '*.fq' \) 2>/dev/null | grep -q .; then
    die "--skip-download set but no zip/FASTQ in $WORKDIR"
  fi
  unpack_zip
else
  [[ -d "$FASTQ_DIR" ]] || die "--fastq-dir is not a directory: $FASTQ_DIR"
  FQ_DIR="$FASTQ_DIR"
fi

PAIR="$(discover_pair "$FQ_DIR")"
R1="${PAIR%%$'\t'*}"
R2="${PAIR#*$'\t'}"
printf 'FASTQ R1: %s\n' "$R1"
printf 'FASTQ R2: %s\n' "$R2"

printf 'Aligning with bwa mem (GRCh38, 150 bp PE expected) ...\n'
bwa mem -t "$THREADS" "$REF_FA" "$R1" "$R2" \
  | samtools sort -@ "$THREADS" -o "$BAM"
samtools index "$BAM"
samtools stats "$BAM" > "$STATS"
samtools idxstats "$BAM" > "$IDXSTATS"

TOTAL="$(sn_field 'raw total sequences:' | tr -cd '0-9')"
UNMAPPED="$(sn_field 'reads unmapped:' | tr -cd '0-9')"
ERROR_RATE="$(sn_field 'error rate:')"
if [[ -z "$TOTAL" ]]; then
  TOTAL="$(awk '{m+=$3; u+=$4} END {print m+u}' "$IDXSTATS" | tr -cd '0-9')"
fi
if [[ -z "$UNMAPPED" ]]; then
  UNMAPPED="$(awk '$1=="*" {print $4; exit}' "$IDXSTATS" | tr -cd '0-9')"
fi
MAPQ30="$(samtools view -c -q 30 "$BAM")"

mapq_frac="n/a"
unmap_frac="n/a"
if [[ -n "${TOTAL:-}" && "${TOTAL:-0}" -gt 0 ]]; then
  mapq_frac="$(awk -v a="$MAPQ30" -v t="$TOTAL" 'BEGIN { printf "%.4f", a/t }')"
  unmap_frac="$(awk -v a="${UNMAPPED:-0}" -v t="$TOTAL" 'BEGIN { printf "%.4f", a/t }')"
fi

mismatch="n/a"
if [[ -n "$ERROR_RATE" ]]; then
  mismatch="$(awk -v e="$ERROR_RATE" 'BEGIN { printf "%.4f", e }')"
fi

peak="$(insert_peak)"
gc="$(gc_percent)"
clip="n/a"
if [[ "$SOFTCLIP" -eq 1 ]]; then
  clip="$(softclip_frac)"
fi

{
  printf 'metric\tthis_run\tpublished_target\n'
  printf 'MAPQ>=30_fraction\t%s\t0.942\n' "$mapq_frac"
  printf 'unmapped_fraction\t%s\t0.003\n' "$unmap_frac"
  printf 'insert_size_peak_bp\t%s\t166\n' "$peak"
  printf 'GC_percent\t%s\t41.2\n' "$gc"
  printf 'mismatch_rate\t%s\t0.0026\n' "$mismatch"
  printf 'softclip_fraction\t%s\t0.0191\n' "$clip"
} | tee "$QC_TABLE"

cat <<EOF

Alignment QC written to ${QC_TABLE}
BAM: ${BAM}

Published Level 1 alignment stats are sanity targets, not pass/fail gates
(bwa mem GRCh38, 150 bp PE; MAPQ>=30 ~94.2%; unmapped ~0.3%;
insert peak ~166 bp; mismatch ~0.26%; soft-clipped ~1.91%; GC ~41.2%).

Mode A complete. This is alignment QC, not a trisomy call.
To run WisecondorX predict you need a synthetic euploid reference.npz:
  ./run_wx.sh --ref /path/to/synthetic_euploid_nipt.npz
EOF
