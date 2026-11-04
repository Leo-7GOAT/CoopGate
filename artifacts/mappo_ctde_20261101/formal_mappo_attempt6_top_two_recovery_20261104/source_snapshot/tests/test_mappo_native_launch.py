import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from mappo_native_launch import command


def test_exact_native_pid_real_cli_arguments_and_existing_packages(tmp_path):
    script=tmp_path/'probe with spaces.py'
    script.write_text('import os,json,argparse,numpy,torch\np=argparse.ArgumentParser();p.add_argument("action",choices=["collect"]);p.add_argument("--tag");a=p.parse_args()\nprint(json.dumps(dict(pid=os.getpid(),action=a.action,tag=a.tag,numpy=numpy.__version__,torch=torch.__version__,torch_file=torch.__file__)))\n')
    child=subprocess.Popen(command(script,'collect','--tag','literal argument with spaces'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    out,err=child.communicate(timeout=30)
    assert child.returncode==0,err
    value=json.loads(out)
    assert value['pid']==child.pid and value['action']=='collect' and value['tag']=='literal argument with spaces'
    assert value['numpy']==np.__version__ and value['torch']==torch.__version__
    assert Path(value['torch_file']).resolve()==Path(torch.__file__).resolve()
