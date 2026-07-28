#!/bin/bash
# Reports which pythons are usable for the experiments
# (needs >= 3.7 plus numpy and scipy).
for cand in "$HOME/anaconda3/envs/pmm/bin/python" \
            "$HOME/miniforge3/envs/pmm/bin/python" \
            /usr/bin/python3 \
            "$HOME/anaconda3/bin/python"; do
  if [ -x "$cand" ]; then
    if "$cand" -c 'import sys; assert sys.version_info>=(3,7); import numpy, scipy' 2>/dev/null; then
      "$cand" -c "import sys, numpy, scipy; print('USABLE: ', '$cand', '| python', sys.version.split()[0], '| numpy', numpy.__version__, '| scipy', scipy.__version__)"
    else
      "$cand" -c "import sys; print('not usable:', '$cand', '| python', sys.version.split()[0], '(needs >=3.7 with numpy+scipy)')" 2>/dev/null || echo "not usable: $cand"
    fi
  fi
done
