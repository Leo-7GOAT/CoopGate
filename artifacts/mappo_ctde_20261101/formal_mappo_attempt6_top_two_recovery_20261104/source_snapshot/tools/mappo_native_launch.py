"""Direct native Python child with the SAME virtualenv packages,without redirector.

Windows venv python.exe can be a launcher whose PID differs from os.getpid().
Use the actual base executable and explicitly process the venv site directory.
No dependencies are installed or changed;runtime provenance stays explicit.
"""
from pathlib import Path
from importlib.util import find_spec
import sys


def command(script,*arguments):
    binary=Path(sys._base_executable).resolve()
    package_directory=Path(find_spec('torch').origin).resolve().parents[1]
    root=Path(__file__).resolve().parents[1]
    if not binary.is_file() or not package_directory.is_dir():raise ValueError('Existing exact Python runtime/packages required')
    bootstrap=('import sys,site,runpy;'
        'package=sys.argv.pop(1);root=sys.argv.pop(1);script=sys.argv.pop(1);sys.argv[0]=script;'
        'site.addsitedir(package);sys.path.insert(0,package);'
        'sys.path.insert(0,root+"/src");sys.path.insert(0,root+"/tools");'
        'runpy.run_path(script,run_name="__main__")')
    return [str(binary),'-S','-c',bootstrap,str(package_directory),str(root),str(Path(script).resolve()),*map(str,arguments)]
