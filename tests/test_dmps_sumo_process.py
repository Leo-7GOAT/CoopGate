from types import SimpleNamespace

from coopgate import dmps_sumo_process as bridge


def test_native_restart_only_closes_owned_connection_and_preserves_load_args(monkeypatch):
    events=[]
    old=SimpleNamespace(close=lambda:events.append('close-owned'))
    new=object()
    class API:
        def start(self,args,**kwargs):events.append(('start',args,kwargs))
        def getConnection(self,label):return new
    monkeypatch.setattr(bridge,'_TraCI',API)
    monkeypatch.setattr(bridge.sumolib,'checkBinary',lambda name:'sumo.exe')
    connection=bridge.ProcessConnection(old,.1)
    connection.load(['--load-state','saved.xml','--begin','.1'])
    assert events[0]=='close-owned'
    assert events[1][1]==['sumo.exe','--load-state','saved.xml','--begin','.1']
    assert connection.original.original is new
    assert connection.native_restarts==1
