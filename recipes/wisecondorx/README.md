# Reproduce the EabhaSeq WisecondorX check

Computational FASTQ, not plasma. Locked published result is 8% fetal fraction
on a synthetic euploid panel. Not claimed at 2% FF.

1. conda env from environment.yml
2. ./run_qc.sh /path/to/GRCh38.fa
   downloads https://eabhaseq.lon1.digitaloceanspaces.com/samples/t21-4m-sample.zip
3. ./run_wx.sh --ref /path/to/synthetic_euploid_nipt.npz
   skipped automatically if no reference is provided

Paper: [bioRxiv DOI placeholder]
Validation: https://eabhaseq.com/validation/details/

The reference and the test BAM must share mapper, reference genome, and file
type. This public recipe is a computational smoke test on labelled synthetic
FASTQ. It does not replace a laboratory plasma QC panel, does not retrain
models, and is not a clinical trisomy call.

## License

This `recipes/wisecondorx/` folder is **MIT** (see `LICENSE`). That is
independent of the parent `cfdna-gen` repository, which uses PolyForm
Noncommercial. Do not treat this recipe as covered by the generator license.

## What this is

EabhaSeq publishes a free 4 million fragment paired-end T21 FASTQ (8% fetal
fraction, live engine) and a locked claim: **WisecondorX v1.2.10 detects
T21/T18/T13 at 8% FF on a synthetic euploid reference panel**. Not claimed at
2% FF. Files are computational, not plasma.

Product: https://eabhaseq.com/

This recipe talks to the public WisecondorX CLI (`wisecondorx convert`,
`newref`, `predict`) plus `bwa` and `samtools`. It does not import private
Python packages, does not load model weights, and does not generate a
50 × 4M-fragment euploid panel on a laptop.

WisecondorX upstream: https://github.com/CenterForMedicalGeneticsGhent/WisecondorX

## Install

```bash
conda env create -f environment.yml
conda activate eabhaseq-wx
chmod +x run_qc.sh run_wx.sh parse_wx_stats.py
```

`environment.yml` pins **wisecondorx 1.2.9** because Bioconda currently
publishes 1.2.9 then 1.3.x (no 1.2.10 tarball). The published claim used
v1.2.10. If your solver can see 1.2.10, change the pin. The CLI is
`wisecondorx convert|newref|predict` in either case.

You must supply **GRCh38 FASTA** (and a `bwa index`). This recipe will not
download a ~3 GB genome. Example public no-alt analysis set (operator
download only):

https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz

```bash
gzip -dc GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz > GRCh38.fa
bwa index GRCh38.fa
```

A 4M PE FASTQ is multi-GB. Do not run this on Colab.

## Mode A — alignment QC (always ships)

```bash
./run_qc.sh /path/to/GRCh38.fa
```

What it does:

1. `curl -L` / `wget` the free zip
2. Unpacks and **discovers** the paired FASTQ names (does not assume `R1.fastq.gz`)
3. `bwa mem` to your GRCh38
4. `samtools sort -o`, `index`, `stats`, `idxstats`
5. Prints a QC table (MAPQ≥30 fraction, unmapped fraction, insert-size peak,
   GC, mismatch rate if available)
6. **Stops.** This is not a trisomy call.

Published Level 1 alignment stats for the generator (sanity targets, not
pass/fail gates):

| metric | target |
|---|---|
| mapper | bwa mem GRCh38, 150 bp PE |
| MAPQ ≥ 30 | ≈ 94.2% |
| unmapped | ≈ 0.3% |
| insert size peak | ≈ 166 bp |
| mismatch rate | ≈ 0.26% |
| soft-clipped | ≈ 1.91% |
| GC | ≈ 41.2% |

Soft-clip is off by default (`--softclip` to scan CIGAR; slow on 4M PE).

If `bwa`, `samtools`, or the FASTA / index is missing, the script prints the
install command and exits. It does not fetch GRCh38.

Useful flags: `--threads N`, `--workdir DIR`, `--skip-download`, `--fastq-dir DIR`.

## Mode B — WisecondorX predict (only if a reference exists)

```bash
./run_wx.sh --ref /path/to/synthetic_euploid_nipt.npz
```

`wisecondorx predict` needs a euploid `reference.npz`. The free zip is one T21
file. It is **not** a reference panel.

Lookup order:

1. `--ref PATH`
2. Environment variable `EABHASEQ_WX_REF`
3. `./ref/synthetic_euploid_nipt.npz`
4. `--ref-url URL`

If none exist, the script prints the exact `newref` command, notes that
EabhaSeq’s published panel was **50 synthetic euploid samples**, and exits 0
with `reference missing; Mode A complete`. It will not call `cfdna_gen` or
HuggingFace and will not generate 50 full-depth samples.

If a reference is present (after you have run Mode A):

```bash
wisecondorx convert sample.bam sample
wisecondorx predict sample.npz $REF out/t21 --bed --plot
```

Then `parse_wx_stats.py` reads `*_statistics.bed` or the v1.2.x
`*_statistics.txt` and prints chromosome 21 / 18 / 13 z. Published synthetic
T21 z at 8% FF was **4.7–6.5**. This run is a pipeline smoke test, not a
clinical call.

### `newref` (documented; not run by default)

WisecondorX docs require ≥50 negative controls from the same mapper /
reference genome / material type.

```bash
wisecondorx newref euploid_npz/*.npz ref/synthetic_euploid_nipt.npz --nipt --binsize 100000
```

This recipe runs that command only if you pass `--build-ref DIR` **and** `DIR`
already contains ≥50 euploid `.npz` files.

```bash
./run_wx.sh --build-ref /path/to/euploid_npz
```

## Files

| file | role |
|---|---|
| `environment.yml` | conda env: wisecondorx, bwa, samtools, curl, unzip |
| `run_qc.sh` | Mode A |
| `run_wx.sh` | Mode B wrapper |
| `parse_wx_stats.py` | stdlib parser for chr 21/18/13 z |
| `Snakefile` | optional thin wrapper (`snakemake --config ref_fa=...`) |

## Optional Snakemake

```bash
snakemake --cores 8 --config ref_fa=/path/to/GRCh38.fa
# Mode B only if wx_ref is set:
snakemake --cores 8 --config ref_fa=/path/to/GRCh38.fa wx_ref=/path/to/synthetic_euploid_nipt.npz
```
