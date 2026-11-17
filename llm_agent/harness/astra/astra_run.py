"""Subscription-backed Astra runs using only the frozen Opus segmentation toolbox."""
import argparse
import contextlib
import csv
import hashlib
import http.server
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path('/Users/kristiansvendsen/MMIV/llm-test')
SUPPORT = Path(__file__).resolve().parent
CLI = Path('/Users/kristiansvendsen/Documents/Codex/2026-11-08/bu/work/astra/cli/node_modules/.bin/codex')
IMAGE = 'sha256:9560b625afffe5228b50f09a1a67795848029f6bc12ee0543087bd0f698a5b8f'
MODEL = 'gpt-6-astra'
TOOLS = {'info','stats','overview','view','review','threshold','grow','components','morph','exclude_box','union','save'}
TIMEOUT = 3600

def dump(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def environment():
    e={k:os.environ[k] for k in ('HOME','USER','LOGNAME','TMPDIR','LANG') if k in os.environ}
    e['PATH']='/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin'
    return e

class RPC:
    def __init__(self, cmd, cwd, env, stderr, log=None):
        self.p=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1,start_new_session=True)
        self.q=queue.Queue(); self.seq=0; self.log=log
        self.reader=threading.Thread(target=self.read,daemon=True)
        self.reader.start()
    def read(self):
        try:
            for line in self.p.stdout:
                if self.log: self.log.write(line); self.log.flush()
                try: self.q.put(json.loads(line))
                except json.JSONDecodeError: pass
        finally: self.q.put(None)
    def send(self, obj):
        self.p.stdin.write(json.dumps(obj)+'\n'); self.p.stdin.flush()
    def notify(self, method, params=None): self.send({'jsonrpc':'2.0','method':method,'params':params or {}})
    def next(self, deadline):
        try: obj=self.q.get(timeout=max(.01,deadline-time.monotonic()))
        except queue.Empty: raise TimeoutError('Wall-clock budget reached')
        if obj is None: raise RuntimeError(f'Process exited: {self.p.poll()}')
        return obj
    def call(self, method, params, deadline):
        self.seq+=1; seq=self.seq
        self.send({'jsonrpc':'2.0','id':seq,'method':method,'params':params})
        while True:
            o=self.next(deadline)
            if o.get('id')==seq:
                if 'error' in o: raise RuntimeError(json.dumps(o['error']))
                return o['result']
            if 'id' in o: raise RuntimeError('Unexpected server request before inference: '+str(o.get('method')))
    def close(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid,signal.SIGTERM)
            try:self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(self.p.pid,signal.SIGKILL);self.p.wait()
        self.reader.join(timeout=5)

def config():
    cfg={'model_catalog_json':str(SUPPORT/'astra_catalog.json'), 'project_doc_max_bytes':0,
         'web_search':'disabled','skills.include_instructions':False,'skills.bundled.enabled':False,
         'include_apps_instructions':False,'include_collaboration_mode_instructions':False,
         'include_permissions_instructions':False,'include_environment_context':False,
         'tools.update_plan.enabled':False,'tools.experimental_request_user_input.enabled':False,
         'agents.enabled':False,'analytics.enabled':False,'mcp_servers':{},'developer_instructions':''}
    for f in ('shell_tool unified_exec apps plugins multi_agent multi_agent_v2 goals view_image image_generation browser_use browser_use_external computer_use memories code_mode code_mode_host tool_suggest skill_search hooks in_app_browser workspace_dependencies sleep_tool default_mode_request_user_input request_permissions_tool enable_request_compression shell_snapshot').split(): cfg['features.'+f]=False
    cfg['features.skip_host_skill_discovery']=True
    return cfg

def run_session(effort, image_dir, out, audit_url=None, connectivity=False):
    out.mkdir(parents=True,exist_ok=True); out.chmod(0o777)
    system,task=(ROOT/'harness/prompt.md').read_text().split('# Task')
    system=system.replace('# System prompt','').strip(); task=task.strip()
    if connectivity: task='Connectivity check only. Reply READY. Do not call any tools.'
    deadline=time.monotonic()+ (240 if audit_url else 90 if connectivity else TIMEOUT)
    container='seg-astra-'+effort+'-'+str(os.getpid())
    docker=['docker','run','-i','--rm','--name',container,'--network','none','--read-only',
            '--tmpfs','/tmp:size=512m','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--pids-limit','256','--memory','4g','--cpus','2','-v',f'{image_dir}:/data:ro','-v',f'{out}:/out',IMAGE]
    tool_rpc=agent=None
    started=time.time()
    try:
        with contextlib.ExitStack() as resources:
            temp=resources.enter_context(tempfile.TemporaryDirectory(prefix='astra-case-'))
            err=resources.enter_context((out/'stderr.log').open('w'))
            transcript=resources.enter_context((out/'transcript.jsonl').open('w'))
            temp=Path(temp); home=temp/'codex';home.mkdir(mode=0o700); cwd=temp/'empty';cwd.mkdir()
            if not audit_url:
                # Host-side login only; never mounted into the image-tools container.
                (home/'auth.json').symlink_to(Path.home()/'.codex/auth.json')
            env=environment();env['CODEX_HOME']=str(home)
            specs=[]
            if not connectivity:
                tool_rpc=RPC(docker,cwd,environment(),err)
                resources.callback(tool_rpc.close)
                tool_rpc.call('initialize',{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'astra-seg-bridge','version':'1'}},deadline)
                tool_rpc.notify('notifications/initialized')
                tool_list=tool_rpc.call('tools/list',{},deadline)['tools']
                assert {t['name'] for t in tool_list}==TOOLS
                dump(out/'tool_schemas.json',tool_list)
                specs=[{'type':'function','name':t['name'],'description':t.get('description',''),'inputSchema':t['inputSchema'],'deferLoading':False} for t in tool_list]
            cfg=config();cfg['model_reasoning_effort']=effort
            provider='openai'
            if audit_url:
                provider='audit';cfg.update({'model_provider':provider,'model_providers.audit.name':'Local request audit','model_providers.audit.base_url':audit_url,'model_providers.audit.wire_api':'responses','model_providers.audit.requires_openai_auth':False})
            cmd=[str(CLI),'app-server']
            for k,v in cfg.items(): cmd+=['-c',k+'='+json.dumps(v)]
            agent=RPC(cmd,cwd,env,err,transcript)
            resources.callback(agent.close)
            agent.call('initialize',{'clientInfo':{'name':'mmiv_astra_experiment','version':'1.0'},'capabilities':{'experimentalApi':True}},deadline)
            agent.notify('initialized')
            thread=agent.call('thread/start',{'model':MODEL,'modelProvider':provider,'allowProviderModelFallback':False,
                'cwd':str(cwd),'ephemeral':True,'approvalPolicy':'never','sandbox':'read-only','environments':[],
                'baseInstructions':system,'developerInstructions':'','dynamicTools':specs,'experimentalRawEvents':True},deadline)
            assert thread['model']==MODEL,thread
            dump(out/'session_config.json',{'model':MODEL,'effort':effort,'image':IMAGE,'config':cfg,'tool_names':sorted(t['name'] for t in specs),'prompt_sha256':digest(ROOT/'harness/prompt.md'),'tools_sha256':digest(ROOT/'harness/tools.py'),'docker_command':docker,'thread_start':thread,'cli':subprocess.check_output([str(CLI),'--version'],text=True).strip()})
            tid=thread['thread']['id']
            agent.call('turn/start',{'threadId':tid,'input':[{'type':'text','text':task}],'model':MODEL,'effort':effort,'environments':[]},deadline)
            calls=0
            while True:
                event=agent.next(deadline);method=event.get('method');params=event.get('params',{})
                if 'id' in event:
                    if method!='item/tool/call' or params.get('tool') not in TOOLS or tool_rpc is None:
                        raise RuntimeError('Unexpected tool/access request: '+str(method)+' '+str(params.get('tool')))
                    name=params['tool'];args=params['arguments']
                    if isinstance(args,str):args=json.loads(args)
                    reply=tool_rpc.call('tools/call',{'name':name,'arguments':args},deadline)
                    content=[]
                    for c in reply.get('content',[]):
                        if c['type']=='text':content.append({'type':'inputText','text':c['text']})
                        elif c['type']=='image':content.append({'type':'inputImage','imageUrl':f"data:{c['mimeType']};base64,{c['data']}"})
                        else: raise RuntimeError('Unexpected MCP output kind '+c['type'])
                    calls+=1
                    agent.send({'jsonrpc':'2.0','id':event['id'],'result':{'success':not reply.get('isError',False),'contentItems':content}})
                    print(f'{effort} {out.name}: tool {calls} {name}',flush=True)
                elif method=='turn/completed':
                    return {'turn':params['turn'],'wall_s':round(time.time()-started),'tool_calls':calls}
    finally:
        if agent: agent.close()
        if tool_rpc: tool_rpc.close()
        if not connectivity:subprocess.run(['docker','rm','-f',container],capture_output=True)

def audit(out, effort="high"):
    requests=[]
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(data);dump(out/'request.json',data)
            self.send_response(400);self.end_headers();self.wfile.write(b'{"error":{"message":"Local audit complete","type":"invalid_request_error"}}')
        def log_message(self,*a):pass
    out.mkdir(parents=True,exist_ok=True)
    server=http.server.HTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        result=run_session(effort,ROOT/'staging/VS-MC-RC-028',out,f'http://127.0.0.1:{server.server_port}/v1')
        print('Audit session:',result)
    finally:server.shutdown()
    req=requests[-1]
    alltools=req.get('tools',[])+[t for item in req.get('input',[]) if item.get('type')=='additional_tools' for t in item.get('tools',[])]
    flat=[]
    for t in alltools:
        flat.extend(t['tools'] if t['type']=='namespace' else [t])
    names=[t['name'] for t in flat]
    print('Model-visible tools:',names)
    assert set(names)==TOOLS and len(names)==len(TOOLS),names
    system,task=(ROOT/'harness/prompt.md').read_text().split('# Task');system=system.replace('# System prompt','').strip();task=task.strip()
    texts=[c['text'] for item in req.get('input',[]) if item.get('type')=='message' for c in item.get('content',[]) if c.get('type')=='input_text']
    print('Prompt messages:',len(texts),[len(t) for t in texts])
    assert texts==[system,task],texts
    assert req['model']==MODEL and req['reasoning']['effort']==effort
    dump(out/'audit_passed.json',{'model':MODEL,'tools':names,'exact_prompt':True,'effort':effort,'image':IMAGE,'request_sha256':digest(out/'request.json')})
    print('AUDIT PASSED')

def run_arm(effort):
    import importlib.util
    spec=importlib.util.spec_from_file_location('opus_run',ROOT/'harness/run.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    audit_file=SUPPORT/f'audit-{effort}/audit_passed.json'
    evidence=json.loads(audit_file.read_text())
    assert evidence['exact_prompt'] and evidence['effort']==effort and evidence['image']==IMAGE
    request=json.loads((audit_file.parent/'request.json').read_text())
    system,task=original.read_prompt()
    messages=[c['text'] for item in request['input'] if item.get('type')=='message' for c in item.get('content',[]) if c.get('type')=='input_text']
    assert messages==[system,task], 'Prompt changed after audit'
    assert digest(ROOT/'harness/tools.py')=='658e5160a4e7a8a749c94c393264b35c2f14a83d1358cbd6331f35232f189de5'
    records=list(csv.DictReader((ROOT/'private/case_key.csv').open()))
    assert len(records)==22 and len({r['patient_id'] for r in records})==22
    arm=ROOT/'runs'/f'astra_{effort}';arm.mkdir(parents=True,exist_ok=True)
    state={'model':MODEL,'effort':effort,'total':22,'completed':0,'state':'running','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    dump(arm/'progress.json',state)
    for index,r in enumerate(records,1):
        pid=r['patient_id'];out=arm/pid
        if (out/'status.json').exists():
            state['completed']=index;continue
        state.update(current_case=pid,position=index);dump(arm/'progress.json',state)
        print(f'[{time.strftime("%H:%M:%S")}] START {effort} {index}/22 {pid}',flush=True)
        started=time.time();result={};error=None
        try:result=run_session(effort,ROOT/'staging'/pid,out)
        except Exception as e:error=f'{type(e).__name__}: {e}'
        status,detail=original.validate(out,ROOT/'staging'/pid/'image.nii.gz')
        detail['unexpected_files']=[f for f in detail.get('unexpected_files',[]) if f not in {'tool_schemas.json','session_config.json'}]
        record={'patient_id':pid,'run_order_case':r['case'],'model':MODEL,'effort':effort,'status':status,**detail,
                'wall_s':round(time.time()-started),'image':IMAGE,'error':error,'result':result,
                'input_sha256':digest(ROOT/'staging'/pid/'image.nii.gz'), 'prompt_sha256':digest(ROOT/'harness/prompt.md'),
                'tools_sha256':digest(ROOT/'harness/tools.py'),'authentication':'ChatGPT subscription','cli_version':'0.167.0'}
        # A transport/setup failure is not a segmentation result. Stop the arm for inspection.
        timed_out=bool(error and error.startswith('TimeoutError:'))
        record['returncode']='timeout' if timed_out else 0 if not error and result.get('turn',{}).get('status')=='completed' else 'error'
        if not timed_out and (error or result.get('turn',{}).get('status')!='completed'):
            dump(out/'infrastructure_error.json',record)
            state.update(state='blocked',error=error or result.get('turn',{}).get('error'));dump(arm/'progress.json',state)
            print(f'BLOCKED {effort} {pid}: {state["error"]}',flush=True)
            raise RuntimeError(f'{effort} stopped on infrastructure/turn error; no automatic rerun')
        dump(out/'status.json',record)
        state['completed']=index;dump(arm/'progress.json',state)
        print(f'[{time.strftime("%H:%M:%S")}] DONE {effort} {index}/22 {pid}: {status}, {record["wall_s"]} seconds',flush=True)
    state.update(state='scoring',current_case=None);dump(arm/'progress.json',state)
    subprocess.run([sys.executable,str(SUPPORT/'astra_score.py'),'--effort',effort],check=True)
    state.update(state='complete',finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'));dump(arm/'progress.json',state)
    print(f'COMPLETE {effort}: 22/22 cases; scored and exported to MMIV reviewer',flush=True)

def launch():
    import fcntl
    from concurrent.futures import ThreadPoolExecutor
    root=ROOT/'runs';root.mkdir(exist_ok=True)
    lock=(root/'astra.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(root/'astra_supervisor.json',{'pid':os.getpid(),'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    def worker(effort):
        arm=root/f'astra_{effort}';arm.mkdir(exist_ok=True)
        with (arm/'progress.log').open('a',buffering=1) as log:
            return subprocess.run([sys.executable,str(Path(__file__).resolve()),'run','--effort',effort],stdout=log,stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=dict(zip(('high','xhigh'),pool.map(worker,('high','xhigh'))))
    dump(root/'astra_supervisor_result.json',results)
    if any(results.values()):sys.exit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','connect','run','launch']);p.add_argument('--out',type=Path);p.add_argument('--effort',choices=['high','xhigh'],default='high');a=p.parse_args()
    if a.command=='audit':audit(a.out.resolve(),a.effort)
    elif a.command=='connect':print(run_session(a.effort,None,a.out.resolve(),connectivity=True))
    elif a.command=='run':run_arm(a.effort)
    else:launch()
