#!/bin/bash
# Diagnoses "ThirdpartyError: Could not find 'fstcompile'" for an MFA conda env.  Usage: bash tools/diagnose_mfa.sh /content/mfa_env
ENV=${1:-/content/mfa_env}
echo "== OpenFst/Kaldi binaries inside $ENV/bin:"; ls "$ENV/bin" 2>/dev/null | grep -E '^(fstcompile|fstcompose|fstarcsort|mfa|compute-mfcc-feats|gmm-align)' || echo "(none found)"
echo "== fstcompile on default PATH:"; which fstcompile || echo "NOT on default PATH"
echo "== fstcompile with env on PATH:"; PATH="$ENV/bin:$PATH" which fstcompile || echo "MISSING from env"
echo "== mfa:"; PATH="$ENV/bin:$PATH" "$ENV/bin/mfa" version 2>&1 | tail -2
cat <<'MSG'

READ THE RESULT:
 A) fstcompile listed in $ENV/bin but "NOT on default PATH"  -> PATH problem only (MFA calls shutil.which). No install needed.
    Fix: run every MFA call with   export PATH=$ENV/bin:$PATH   (preprocess.py --stage align already does this).
 B) fstcompile "MISSING from env" -> openfst/kaldi binaries really absent. One targeted install:
      conda install -y -p $ENV -c conda-forge openfst     (or: micromamba install -y -p $ENV -c conda-forge openfst)
    then re-run this script to confirm.
MSG
