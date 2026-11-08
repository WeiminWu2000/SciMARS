"""Seven-role public-only team for AppWorld's stateful Python/API environment."""
import ast
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys

from mah2.gateway import BudgetExceeded
from mah2.dsstar.harness import initial

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = '''You operate AppWorld, a simulated world, using Python code and its documented APIs.
The task is completed by changing app state or answering via apis.supervisor.complete_task.
Python variables persist between execute calls. Use print to inspect responses.
The variable apis already exists; do not import apis. Consult API documentation
before using a signature or assuming response fields. Handle pagination.
Discover APIs with apis.api_docs.show_app_descriptions(),
apis.api_docs.show_api_descriptions(app_name=...), and
apis.api_docs.show_api_doc(app_name=..., api_name=...).
Use the documented supervisor APIs for account information. Do not invent API signatures.
Only public APIs are permitted. Do not access host files, ground truth, evaluators,
database internals, private APIs, or network services. No import appworld.
Each role receives the same task; handoff fields and role procedures may vary.
Return one JSON object as requested. Runtime success alone is not task correctness.'''
# Restates the public supervisor.complete_task API documentation (no grader information).
COMPLETION_CONTRACT = (
    'Completion contract (public supervisor.complete_task documentation): pass answer if and only if the '
    'task asks for information, and make it only the requested value (a number, yes/no, or a specific '
    'name/entity), never a sentence, summary or status report. If the task asks you to perform actions '
    'rather than answer a question, call apis.supervisor.complete_task() with no answer once the actions '
    'are verified.')
RETRY_NOTE = ('Your previous reply was not one complete JSON object ({error}). Reply again with one complete, '
              'shorter JSON object containing only the requested fields.')


def system_prompt(config):
    return SYSTEM + ('\n' + COMPLETION_CONTRACT if config.get('completion_contract') else '')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object')
    return result


class World:
    def __init__(self, task, experiment, directory, config):
        self.log = open(Path(directory) / 'world_stderr.log', 'a')
        self.env = dict(os.environ, APPWORLD_ROOT=str(Path(os.environ.get('APPWORLD_ROOT', ROOT / 'appworld_data')).resolve()))
        self.p = subprocess.Popen(
            [os.environ.get('MAH2_APPWORLD_PYTHON', str(ROOT / '.venv-appworld/bin/python')), '-m', 'mah2.appworld_worker',
             '--task', task, '--experiment', experiment], cwd=ROOT, env=self.env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1)
        try:
            self.info = self.read(300)
        except Exception:
            self.close()
            raise

    def read(self, timeout=90):
        if not select.select([self.p.stdout], [], [], timeout)[0]:
            raise TimeoutError('AppWorld worker response timed out')
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('AppWorld worker exited; inspect world_stderr.log')
        return json.loads(line)

    def execute(self, code):
        ast.parse(code)
        self.p.stdin.write(json.dumps({'code': code}) + '\n')
        self.p.stdin.flush()
        return self.read()

    def close(self):
        if self.p.poll() is None:
            try:
                self.p.stdin.write('{"close":true}\n'); self.p.stdin.flush()
                self.p.wait(timeout=15)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.p.kill(); self.p.wait()
        self.log.close()


class Team:
    def __init__(self, gateway, config, directory, task, harness=None, workflow=''):
        self.g, self.c, self.root, self.task = gateway, config, Path(directory), task
        self.h = harness or initial()
        self.workflow = workflow
        self.root.mkdir(parents=True, exist_ok=False)
        self.events = []
        self.state = dict(plans='', code='', execution='', verification='', routing='')

    def event(self, **kw):
        event = dict(id=f'e{len(self.events):04d}', **kw)
        self.events.append(event)
        with (self.root / 'trace.jsonl').open('a') as f:
            f.write(json.dumps(event) + '\n')
        return event

    def ask(self, role, instruction):
        fields = {
            'analyzer': (), 'planner': ('execution', 'plans', 'routing'),
            'coder': ('plans', 'code', 'execution'),
            'verifier': ('plans', 'code', 'execution'),
            'router': ('plans', 'execution'),
            'debugger': ('code', 'execution'),
            'finalizer': ('plans', 'execution', 'verification'),
        }[role]
        handoff = {k: self.state.get(k) for k in fields}
        coord = self.h['coordination']
        if ((role == 'planner' and coord['planner_receives_diagnosis']) or
                (role == 'router' and coord['router_receives_diagnosis'])):
            handoff['verification'] = self.state['verification']
        instructions = []
        for edge, contract in coord['contracts'].items():
            sender, receiver = edge.split('.')
            if role == sender:
                instructions.append(contract['sender_instruction'])
            if role == receiver:
                handoff.update({k:self.state.get(k) for k in contract['fields']})
                instructions.append(contract['receiver_instruction'])
        prompt = dict(role=role, task=self.info, instruction=instruction,
                      role_procedure=self.h['roles'][role], workflow=self.workflow,
                      handoff=handoff, contracts=instructions,
                      recent_public_interactions=[
                          dict(id=e['id'],code=e['code'],output=e['output'][-6000:])
                          for e in self.events if e['kind']=='execution'][-6:])
        self.event(kind='handoff', receiver=role, prompt=prompt)
        return self.request_reply(role, prompt, self.token_cap(role))

    def token_cap(self, role):
        """Code-writing roles may use execution_output_tokens (long programs were cut off at 2048)."""
        default = self.c.get('max_output_tokens', 2048)
        if role in ('analyzer', 'coder', 'debugger', 'finalizer'):
            return self.c.get('execution_output_tokens', default)
        return default

    def request_reply(self, role, prompt, max_tokens, parse=parse_json):
        """One role completion. With recoverable_step_errors, an unparseable reply (usually cut off at
        max_tokens) is re-requested once instead of aborting the rollout; the retry is paid and logged."""
        messages = [{'role': 'system', 'content': system_prompt(self.c)},
                    {'role': 'user', 'content': json.dumps(prompt)}]
        for attempt in range(2 if self.c.get('recoverable_step_errors') else 1):
            response = self.g.request({'messages': messages, 'max_tokens': max_tokens,
                                       'response_format': {'type': 'json_object'}})
            text = response['choices'][0]['message'].get('content') or ''
            self.event(kind='role_output', role=role, text=text)
            try:
                return parse(text)
            except (ValueError, json.JSONDecodeError) as exc:
                if attempt or not self.c.get('recoverable_step_errors'):
                    raise
                self.event(kind='reply_retry', role=role, error=f'{type(exc).__name__}: {exc}')
                messages = messages + [{'role': 'user', 'content': RETRY_NOTE.format(error=exc)}]

    def run(self):
        experiment = 'mah2/' + hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:24]
        write_json(self.root/'harness.json', self.h)
        programs = []
        completed = False
        error = None
        world = None

        def execute(reply):
            nonlocal completed
            code = reply.get('code')
            if self.c.get('recoverable_step_errors'):
                # Malformed code is an observation for the debugger, not the end of the rollout.
                problem = None
                if not isinstance(code, str):
                    problem = 'Host rejected reply: "code" must be a string of Python.'
                else:
                    try:
                        ast.parse(code)
                    except SyntaxError as exc:
                        problem = f'Traceback (host syntax check, nothing executed):\nSyntaxError: {exc}'
                if problem:
                    code = code if isinstance(code, str) else json.dumps(code)
                    self.state.update(code=code, execution=problem)
                    self.event(kind='execution', code=code, output=problem, completed=completed, executed=False)
                    return dict(output=problem, completed=completed, error=True)
            if not isinstance(code, str):
                raise ValueError('Execution role must return string code')
            programs.append(code)
            result = world.execute(code)
            self.state.update(code=code, execution=result['output'][-14000:])
            completed = result['completed']
            self.event(kind='execution', code=code, output=result['output'], completed=completed)
            return result

        try:
            # Worker start-up is inside the rollout's error handling: a slow or failed start (seen
            # under GPFS load) fails this rollout only, never the whole method-task run.
            for attempt in range(2):
                try:
                    world = World(self.task, experiment, self.root, self.c)
                    break
                except (TimeoutError, RuntimeError) as exc:
                    self.event(kind='world_start_failure', attempt=attempt, error=f'{type(exc).__name__}: {exc}')
                    if attempt:
                        raise
            self.info = world.info
            self.state['question'] = self.info['instruction']
            execute(self.ask('analyzer', 'Inspect relevant API documentation and public state. Return {"code": "Python"}. Do not mutate app state yet.'))
            for step in range(self.c.get('max_team_steps', 8)):
                if completed:
                    break
                self.state['plans'] = self.ask('planner', 'Plan the next small executable step using observed state. Return {"plan": "..."}.')['plan']
                result = execute(self.ask('coder', 'Implement the next step, printing needed observations. Return {"code": "Python"}. Call complete_task only after checking the requested result; include the answer for question tasks.'))
                if completed:
                    break
                if result.get('error') or 'Traceback' in result['output']:
                    result = execute(self.ask('debugger', 'Repair the failed step using the error and API docs. Do not repeat successful state-changing actions. Return {"code":"Python"}.'))
                if completed:
                    break
                review = self.ask('verifier', 'Assess progress using public observations. Do not infer success from lack of errors. Return {"ready":true/false,"diagnosis":"remaining requirements and observed evidence"}.')
                self.state['verification'] = json.dumps(review) if self.h['coordination']['verifier_diagnosis'] else str(review.get('ready', False))
                route = self.ask('router', 'Choose continue or finalize based on evidence. Return {"route":"continue" or "finalize","reason":"..."}.')
                self.state['routing'] = json.dumps(route)
                if route.get('route') == 'finalize':
                    execute(self.ask('finalizer', 'Finalize only if the task is satisfied, using apis.supervisor.complete_task with any required answer. Otherwise inspect the missing evidence. Return {"code":"Python"}.'))
            if not completed:
                error = 'Team step limit reached without completion'
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            self.event(kind='rollout_failure', error=error)
        finally:
            if world is not None:
                world.close()
        # This is a trajectory, not a standalone replay script: observations
        # between blocks can determine subsequent actions.
        program = '\n\n'.join(f'# interaction {i}\n{x}' for i,x in enumerate(programs))
        (self.root/'submission.py').write_text(program)
        outcome = dict(runnable=bool(programs), completed=completed, error=error,
                       program_sha256=hashlib.sha256(program.encode()).hexdigest(),
                       experiment=experiment, task_id=self.task, official_success=None)
        write_json(self.root/'outcome.json', outcome)
        return outcome


def grade_selected(outcome, destination):
    """Called only after the final selection and its hash are saved."""
    env = dict(os.environ, APPWORLD_ROOT=str(Path(os.environ.get('APPWORLD_ROOT', ROOT/'appworld_data')).resolve()))
    result = subprocess.run([os.environ.get('MAH2_APPWORLD_PYTHON', str(ROOT/'.venv-appworld/bin/python')), '-m',
                             'mah2.appworld_worker', '--grade', '--task', outcome['task_id'],
                             '--experiment', outcome['experiment']], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=600)
    (Path(destination)/'grading_stderr.log').write_text(result.stderr)
    result.check_returncode()
    report = json.loads(result.stdout)
    write_json(Path(destination)/'grade.json', report)
    return report
