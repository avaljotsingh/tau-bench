import json
import os
import ast
import textwrap
from termcolor import colored
from llmagent import LLMAgent
from libgen_utils import *
from tau_bench.trapi_infer import completion, gen_completion, model_dump

class FunctionSuggestionAgent(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def suggest_funcs(self, tasks, library):
        system_prompt = f'''
You are an expert at generating functions from tasks.
You will be given a list of conversations between a user and an assistant.
Your task is to propose high-level functions that are commonly used in solving the user's requests.
You are also given the current library of already defined functions.
You need to suggest one function that can be added to this library and can be used in most of the observed tasks.
Output only the name of the function, its arguments and the description in the following json format:
{{
    "name": <name of the function>,
    "arguments": <arguments of the function>,
    "description": <description of the function>
}}
'''
        user_message = "\n".join(f"Conversation: {task['traj']}" for task in tasks)
        user_message += f"\nCurrent Library: {library}"
        response = gen_completion(
            model=self.model_name,
            custom_llm_provider=os.environ.get("LIBGEN_PROVIDER", "openai"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format="json_object",
        )
        msg = model_dump(response.choices[0].message)
        content = msg["content"].strip()
        return json.loads(content)

class LibRankerAgent(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def rank_funcs(self, funcs):
        system_prompt = f'''
You are an expert in creating a library of functions.
You will be given a list of functions that are are defined.
You are also given a list of functions that we wish to define. 
However, while defining a function, you may need the other one. 
So, your task is to rank the functions in such a way that to define a function with a lower rank we do not need the function with a higher rank.
So, basically give a topological sort.
Output in the following format:
function_1
function_2
...
Just output the function names in the correct order.       
Do not output any explanation or anything else. 
If there are multiple functions that can be defined, prefer the one that has more utility.
'''
        response = gen_completion(
            model=self.model_name,
            custom_llm_provider=os.environ.get("LIBGEN_PROVIDER", "openai"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(funcs)},
            ],
        )
        msg = model_dump(response.choices[0].message)
        return msg["content"].strip()
    

class FuncDefinitionAgent(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def define_func(self, library, new_func, tasks):
        system_prompt = f'''
You are an expert in writing new functions. 
You will be given a list of conversations between a user and an assistant.
There is a library of functions that can be used to solve the queries.
However, the user has suggested a new function that would be helpful to add to the current set.
Your task is to define the new function.
You can use the functions in the current set.
You can use the queries and the steps taken to solve them to understand what the function does.    
Do not use any function that is not in the current set.
Remember the all the inputs to the function must be string. You can typcast them internally if you want. 
For example, if you want an input parameter x to be a string, you should expect that the user will provide it as a string.
You can internally say x = str(x).
Regardless of the types, in the function definition, do not give any type hints.
Also, make sure that your function has a doc string.
Output a JSON object in the following fomat:
{{
    "new_function": <new_function>,
    "explanation": <explanation>
}}
'''
        user_message = f'Current available functions: {library}\nNew function: {new_func}\nSolved Tasks: {tasks}'
        response = gen_completion(
            model=self.model_name,
            custom_llm_provider=os.environ.get("LIBGEN_PROVIDER", "openai"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format="json_object"
        )
        msg = model_dump(response.choices[0].message)
        content = msg["content"].strip()
        parsed = json.loads(content)
        # print(parsed)
        return parsed['new_function']

class DocStringGenerator(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_docstring(self, func):
        system_prompt = f'''
You are an expert in writing docstrings for functions.
You will be given a function along with a docstring.
Your task is to change the docstring for the function to the following format:
{{
  "type": "function",
  "function": {{
    "name": "get_orders_by_status",
    "description": "<full free-text docstring here>",
    "parameters": {{
      "type": "object",
      "properties": {{}},
      "required": []
    }}
  }}
}}
Output only the format in the following json format:
{{
    "explanation": <explanation of the changes>
    "function": <updated function with docstring>,
}}
'''
        user_message = f'Function {func}'
        response = gen_completion(
            model=self.model_name,
            custom_llm_provider=os.environ.get("LIBGEN_PROVIDER", "openai"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format="json_object"
        )
        msg = model_dump(response.choices[0].message)
        content = msg["content"].strip()
        parsed = json.loads(content)
        return parsed['function']

class FuncCorrector(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def correct_function(self, func):
        system_prompt = f'''
You are an expert at predicting problems with functions.
The user generated a function to be used by an LLM agent.
Although the function is conceptually fine, since it is called by an LLM agent, it may be incorrect.
The LLM may not call the function with the correct arguments or expected types.
Your task is to predict such problems and correct the function by trying to convert the argument into the required format first.
In case you cannot handle things, make sure top return a proper error message so that someone can look at the logs and interpret the error.
You need to make sure that the docstring is in a proper format as before.
Output a JSON object in the following fomat:
{{
    "new_function": <new_function>,
    "explanation": <explanation>
}}
'''
        response = gen_completion(
            model=self.model_name,
            custom_llm_provider=os.environ.get("LIBGEN_PROVIDER", "openai"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'{func}'},
            ],
            response_format="json_object"
        )
        msg = model_dump(response.choices[0].message)
        content = msg["content"].strip()
        parsed = json.loads(content)
        return parsed["new_function"]


class FuncCorrectorFromTrajectories(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def correct_function(self, old_library, new_func, new_trajectory, new_func_def_history):
        system_prompt = f'''
You are an expert at debugging and improving functions.
The user had to solve a task and they had access to a list of tools they could use.
Later, there was an addition to the set of functions, and the user solved the task again.
However, it did not improve the results because the function was not defined properly.
Most errors correcspond to the argument parsing. The input was expected to be of a format but it was not.
You can correct these things by adding some additional code that converts the arguments into the required type.
Your job is to look at the trajectories and improve the newly added function.
Output a JSON object in the following fomat:
{{
    "explanation": <explanation>,
    "new_function": <new_function>,
}}
Only change the things that caused the mistake. Do not predict any new changes.
'''
        messages = [{"role": "system", "content": system_prompt}] + new_func_def_history + [{"role": "user", "content": f'Old Library: {old_library}\nNew Function: {new_func}\n\nNew Trajectory: {new_trajectory}'}]
        response = gen_completion(
            model=self.model_name,
            custom_llm_provider=os.environ.get("LIBGEN_PROVIDER", "openai"),
            messages=messages,
            response_format="json_object"
        )
        msg = model_dump(response.choices[0].message)
        content = msg["content"].strip()
        parsed = json.loads(content)
        # print(parsed)
        # ljkdf
        return parsed["new_function"], parsed["explanation"]


# def get_tool_description(tool):
#     """Parses tool metadata and returns a Func object."""
#     args = []

#     for param_name, param_schema in tool.inputSchema['properties'].items():
#         param_type = param_schema.get('type')
#         default = param_schema.get('default', None)
#         args.append((param_name, param_type, default))

#     return Func(
#         name=tool.name,
#         args=args,
#         description=tool.description
#     )


# def extract_docstring_from_function_string(function_str: str) -> str:
#     try:
#         tree = ast.parse(function_str)
#         for node in tree.body:
#             if isinstance(node, ast.FunctionDef):
#                 return ast.get_docstring(node)
#     except SyntaxError as e:
#         print(f"Syntax error while parsing: {e}")
#     return None

# def is_docstring_json(docstring: str) -> bool:
#     if not docstring:
#         return False
    
#     def clean_trailing_commas(s):
#         # Remove trailing commas before } or ]
#         return re.sub(r',(\s*[}\]])', r'\1', s)
    
#     doc_cleaned = clean_trailing_commas(docstring)
#     try:
#         json.loads(doc_cleaned)
#         return True
#     except json.JSONDecodeError:
#         return False


class FunctionSynthesisAgent(LLMAgent):
    """Deterministic-assembly tool generator.

    The model only fills three narrow, validated slots (description, parameter
    schema, and the function body). This module owns the structure: the `def`
    line, the JSON tool-schema docstring, indentation, and (via create_file) the
    `@mcp.tool()` wrapper. This removes the failure mode where the model returns
    a tool-schema object in place of runnable Python source.
    """

    def synthesize(self, suggested_func, library, tasks, failure_context=None):
        system_prompt = '''
You are implementing a new composite tool for a Python MCP server used by an LLM agent.
You are given:
- A proposed function (its intended name and purpose).
- The library of base tools already available (names and parameters). These base tools are
  ordinary Python functions already in scope; your body may call them directly by name.
- Example task conversations (and, when present, an analysis of WHY the agent failed the
  task). When a failure analysis is given, implement the tool so it directly addresses that
  failure mode - i.e. so a single call returns exactly what the agent needed but failed to get.

Write the IMPLEMENTATION of the proposed function and output a STRICT JSON object with exactly:
{
  "description": "<concise free-text description of what the function does>",
  "parameters": {
    "type": "object",
    "properties": { "<arg_name>": {"type": "string", "description": "<desc>"} },
    "required": ["<arg_name>"]
  },
  "body": "<Python statements for the function body ONLY>"
}

Hard rules for "body":
- Do NOT include the `def` line, decorators, or a docstring. Only the statements inside the function.
- Every parameter is passed as a string by the agent. Typecast inside the body as needed
  (e.g. `import json; ids = json.loads(ids)` for a list, `n = int(n)` for a number).
- Do not use type hints.
- Call base tools by their exact names. Do not invent tools that are not in the library.
- Return a JSON-serializable result. On bad input, `return {"error": "<message>"}` rather than raising.
- Write the statements at the left margin (no leading indentation); they will be indented on assembly.
- Each property name in "parameters" must be a valid Python identifier matching a parameter you use.
'''
        user_message = (
            f"Proposed function: {json.dumps(suggested_func)}\n\n"
            "Base tool library:\n" + "\n".join(str(t) for t in library) + "\n\n"
            "Example tasks:\n" + "\n".join(f"Conversation: {t.get('traj')}" for t in tasks)
        )
        if failure_context:
            user_message += f"\n\nFailure analysis (the tool must address this):\n{failure_context}"
        response = gen_completion(
            model=self.model_name,
            custom_llm_provider=os.environ.get("LIBGEN_PROVIDER", "openai"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format="json_object",
        )
        msg = model_dump(response.choices[0].message)
        return json.loads(msg["content"].strip())

    def correct_body(self, library, current_source, failed_trajectory, history):
        system_prompt = '''
You previously implemented a composite tool, but running it on a task produced an error.
Fix ONLY the function body so the error no longer occurs. Most errors are argument parsing
(the agent passed a string where a list/number was expected) or calling a base tool incorrectly.
Keep the same behavior otherwise. Output a STRICT JSON object:
{
  "explanation": "<what was wrong and how you fixed it>",
  "body": "<corrected Python statements for the function body ONLY, no def line, no docstring>"
}
Rules: parameters arrive as strings (typecast inside); no type hints; call base tools by name;
return {"error": "<message>"} on bad input; write statements at the left margin (no leading indent).
'''
        user_message = (
            "Base tool library:\n" + "\n".join(str(t) for t in library) + "\n\n"
            f"Current function source:\n{current_source}\n\n"
            f"Failed trajectory:\n{failed_trajectory}"
        )
        messages = (
            [{"role": "system", "content": system_prompt}]
            + (history or [])
            + [{"role": "user", "content": user_message}]
        )
        response = gen_completion(
            model=self.model_name,
            custom_llm_provider=os.environ.get("LIBGEN_PROVIDER", "openai"),
            messages=messages,
            response_format="json_object",
        )
        msg = model_dump(response.choices[0].message)
        parsed = json.loads(msg["content"].strip())
        return parsed["body"], parsed.get("explanation", "")


def _build_tool_schema(name, description, parameters):
    """Build the canonical tool-schema dict that becomes the function docstring."""
    if not isinstance(parameters, dict):
        parameters = {}
    props = parameters.get("properties", {}) or {}
    required = parameters.get("required")
    if required is None:
        required = list(props.keys())
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or "",
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def _assemble_function_source(name, schema, param_names, body):
    """Deterministically assemble runnable Python: def + JSON-schema docstring + body."""
    doc = json.dumps(schema, indent=2)
    sig = ", ".join(param_names)
    body = (body or "").strip("\n")
    if not body.strip():
        body = 'return {"error": "not implemented"}'
    indented_body = textwrap.indent(body, "    ")
    return f'def {name}({sig}):\n    """\n{doc}\n    """\n{indented_body}\n'


def _is_valid_function_source(source):
    """Source must parse AND its docstring must be the JSON tool-schema."""
    if not isinstance(source, str):
        return False
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    doc = extract_docstring_from_function_string(source)
    return is_docstring_json(doc)


def _parse_function_source(source):
    """Recover (name, param_names, schema, body) from an assembled function source."""
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    name = fn.name
    params = [a.arg for a in fn.args.args]
    doc = ast.get_docstring(fn)
    schema = json.loads(doc) if doc else None
    has_doc = bool(fn.body) and isinstance(fn.body[0], ast.Expr) and isinstance(
        getattr(fn.body[0], "value", None), ast.Constant
    )
    body_nodes = fn.body[1:] if has_doc else fn.body
    body = "\n".join(ast.unparse(n) for n in body_nodes)
    return name, params, schema, body


def correct_func_from_traj(old_library, new_func, new_trajectory, new_func_def_history):
    """Re-synthesize only the body of an existing tool to fix a failing trajectory.

    Name, signature, and JSON-schema docstring are preserved deterministically; the
    model supplies only the corrected body. Returns (source, explanation).
    """
    synth_agent = FunctionSynthesisAgent()
    try:
        name, param_names, schema, _ = _parse_function_source(new_func)
    except Exception:
        return new_func, "Could not parse prior function source; left unchanged."
    body, explanation = synth_agent.correct_body(
        old_library, new_func, new_trajectory, new_func_def_history
    )
    source = _assemble_function_source(name, schema, param_names, body)
    if not _is_valid_function_source(source):
        return new_func, explanation
    return source, explanation


def get_new_func(tasks, old_library, verbose=True):
    """Propose and deterministically assemble a new composite tool.

    Returns (function_name, source) where source is a runnable `def` whose docstring
    is the JSON tool-schema. Returns (name, None) if synthesis cannot be validated.
    """
    suggest_agent = FunctionSuggestionAgent()
    synth_agent = FunctionSynthesisAgent()

    suggested_func = suggest_agent.suggest_funcs(tasks, old_library)
    name = suggested_func["name"]
    if verbose:
        print(colored(f"Suggested function name: {name}", "blue"))

    for attempt in range(3):
        try:
            impl = synth_agent.synthesize(suggested_func, old_library, tasks)
        except Exception as e:
            if verbose:
                print(colored(f"Attempt {attempt + 1}: synthesis call failed: {e}", "yellow"))
            continue
        description = impl.get("description") or suggested_func.get("description", "")
        parameters = impl.get("parameters", {})
        body = impl.get("body", "")
        props = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        param_names = [p for p in props.keys() if isinstance(p, str) and p.isidentifier()]
        schema = _build_tool_schema(name, description, parameters)
        source = _assemble_function_source(name, schema, param_names, body)
        if _is_valid_function_source(source):
            if verbose:
                print(colored(f"Synthesized function definition:\n{source}", "green"))
            return name, source
        if verbose:
            print(colored(f"Attempt {attempt + 1}: invalid synthesis, retrying", "yellow"))
    return name, None


class FailureFunctionSuggestionAgent(LLMAgent):
    """Propose a composite tool that targets a specific task FAILURE.

    Unlike FunctionSuggestionAgent (which mines patterns from solved trajectories),
    this conditions on a failed trajectory plus a why-it-failed analysis, so the
    proposed tool aims to turn that failure into a success.
    """

    def suggest_from_failure(self, failed_trajectory, failure_reason, library):
        system_prompt = '''
You are an expert at improving an LLM agent's tool library by learning from its FAILURES.
You are given: a conversation where the agent FAILED a task, an analysis of WHY it failed,
and the current library of base tools. Propose ONE new high-level composite tool that, had it
existed, would most plausibly have let the agent avoid this failure (e.g. by returning the exact
information it needed in one call, or by enforcing a step it skipped/got wrong).
Output ONLY strict JSON:
{ "name": "<snake_case_name>", "arguments": ["<arg>", ...], "description": "<what it does and how it fixes the failure>" }
'''
        user_message = (
            f"Failed conversation:\n{failed_trajectory}\n\n"
            f"Reason for failure:\n{failure_reason}\n\n"
            "Current base tool library:\n" + "\n".join(str(t) for t in library)
        )
        response = gen_completion(
            model=self.model_name,
            custom_llm_provider=os.environ.get("LIBGEN_PROVIDER", "openai"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format="json_object",
        )
        msg = model_dump(response.choices[0].message)
        return json.loads(msg["content"].strip())


def get_new_func_from_failure(failed_trajectory, failure_reason, old_library, verbose=True):
    """Failure-driven deterministic generation.

    Propose a tool targeting the failure, then synthesize runnable source for it via
    the same deterministic assembly + validation as get_new_func. The failure analysis
    is threaded into synthesis so the body addresses the failure mode directly.
    Returns (name, source) or (name, None) if synthesis can't be validated.
    """
    suggest_agent = FailureFunctionSuggestionAgent()
    synth_agent = FunctionSynthesisAgent()

    suggested_func = suggest_agent.suggest_from_failure(failed_trajectory, failure_reason, old_library)
    name = suggested_func["name"]
    if verbose:
        print(colored(f"[failure-driven] suggested: {name}", "blue"))

    failure_context = f"Why the agent failed: {failure_reason}"
    pseudo_tasks = [{"traj": failed_trajectory}]
    for attempt in range(3):
        try:
            impl = synth_agent.synthesize(suggested_func, old_library, pseudo_tasks, failure_context=failure_context)
        except Exception as e:
            if verbose:
                print(colored(f"Attempt {attempt + 1}: synthesis call failed: {e}", "yellow"))
            continue
        description = impl.get("description") or suggested_func.get("description", "")
        parameters = impl.get("parameters", {})
        body = impl.get("body", "")
        props = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        param_names = [p for p in props.keys() if isinstance(p, str) and p.isidentifier()]
        schema = _build_tool_schema(name, description, parameters)
        source = _assemble_function_source(name, schema, param_names, body)
        if _is_valid_function_source(source):
            if verbose:
                print(colored(f"[failure-driven] synthesized {name}", "green"))
            return name, source
        if verbose:
            print(colored(f"Attempt {attempt + 1}: invalid synthesis, retrying", "yellow"))
    return name, None
