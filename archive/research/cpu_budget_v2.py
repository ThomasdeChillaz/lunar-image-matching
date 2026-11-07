"""Avoid hour-long BF16 software emulation on standard free CPU runners."""
import argparse
import json
from pathlib import Path
import platform
import sys


def inspect_cpu(content=None):
    if content is None:content=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    models=sorted({s.split(':',1)[1].strip() for s in content.splitlines() if s.startswith('model name')})
    flags=set()
    for line in content.splitlines():
        if line.startswith(('flags','Features')):flags.update(line.split(':',1)[1].split())
    native=bool(flags&{'avx512_bf16','amx_bf16','bf16'})
    return dict(machine=platform.machine(),cpu_models=models,native_bf16=native,
        bf16_flags=sorted(flags&{'avx512_bf16','amx_bf16','bf16'}),
        scope='Author BF16 CPU profile within the finite free-compute budget. No model, precision or configuration substitution.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    doc=inspect_cpu();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc),flush=True)
    if not doc['native_bf16']:sys.exit('Resource profile unavailable: this free CPU has no reported native BF16 feature; retry the unchanged job on another standard free runner.')
