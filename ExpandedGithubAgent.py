import os
import json
import asyncio
import shutil
import stat
import errno
import sys
import re
import ast
from collections import Counter
from git import Repo
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- TREE-SITTER IMPORTS ---
try:
    import tree_sitter
    # Import core classes directly from the binding
    from tree_sitter import Parser, Query
    from tree_sitter_languages import get_language
    HAS_TREESITTER = True
except ImportError:
    print("❌ CRITICAL: 'tree-sitter' or 'tree-sitter-languages' not found.")
    print("👉 Please run: pip install tree-sitter tree-sitter-languages")
    sys.exit(1)

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"

# Graph Configuration
MAX_RECURSION_DEPTH = 4       
MAX_CONCURRENCY = 50 
UBIQUITOUS_THRESHOLD = 15     
CACHE_FILE = "symbol_graph_cache.json"

# Paths
TEMP_DIR = "./temp_session_data"
KNOWLEDGE_BASE_FILE = "symbol_graph.json"

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not found in .env. Please set it.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Helper: Force Delete Read-Only Files ---
def handle_remove_readonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

# --- Helper: Robust Retry Wrapper ---
async def safe_chat_completion(model, messages, response_format=None, retries=3):
    base_delay = 2
    for attempt in range(retries):
        try:
            kwargs = {"model": model, "messages": messages, "timeout": 45}
            if response_format: kwargs["response_format"] = response_format
            return await client.chat.completions.create(**kwargs)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait = base_delay * (2 ** attempt)
                print(f"   ⚠️ Rate limit. Pausing {wait}s...", flush=True)
                await asyncio.sleep(wait)
            elif "json" in str(e).lower():
                print(f"   ⚠️ JSON Error. Retrying...", flush=True)
            else:
                if attempt == retries - 1: raise e
    raise Exception("Exceeded max retries.")

# --- Cleanup ---
def perform_cleanup():
    print("\n🧹 Cleaning up temporary files...")
    if os.path.exists(TEMP_DIR):
        try: shutil.rmtree(TEMP_DIR, onerror=handle_remove_readonly)
        except: pass
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        try: os.remove(KNOWLEDGE_BASE_FILE)
        except: pass
    print("✅ Cleanup complete.")

# --- 1. Universal File Loader ---

def is_valid_file(filename):
    ALWAYS_KEEP_NAMES = {
        'dockerfile', 'makefile', 'gemfile', 'jenkinsfile', 'procfile', 
        'requirements.txt', 'package.json', 'cargo.toml', 'go.mod', 'pom.xml'
    }
    ALLOWED_EXTENSIONS = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', 
        '.cpp', '.cc', '.c', '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.sh',
        '.css', '.scss', '.sass', '.less'
    }
    name = os.path.basename(filename).lower()
    ext = os.path.splitext(filename)[1].lower()
    return name in ALWAYS_KEEP_NAMES or ext in ALLOWED_EXTENSIONS

def read_universal_text(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f: return f.read()
    except: return ""

async def async_read_file(path, relative_path):
    return await asyncio.to_thread(read_universal_text, path)

async def handle_github_repo(url, source_id):
    if not url: return None, None
    clean_name = url.split("/")[-1].replace(".git", "")
    if not clean_name: clean_name = f"repo_{source_id}"
    repo_path = os.path.join(TEMP_DIR, f"{clean_name}_{source_id}")
    
    print(f"🔄 Cloning {clean_name}...")
    try:
        await asyncio.to_thread(Repo.clone_from, url, repo_path)
        return repo_path, clean_name
    except Exception as e:
        print(f"❌ Git Clone Failed for {url}: {e}")
        return None, None

async def ingest_sources(github_inputs: str):
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR, onerror=handle_remove_readonly)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    tasks = []
    git_urls = [s.strip() for s in github_inputs.split(',') if s.strip()]
    
    dir_tasks = [handle_github_repo(url, i) for i, url in enumerate(git_urls)]
    repo_results = await asyncio.gather(*dir_tasks)
    
    multi_repo_data = {}
    read_tasks = []
    
    for repo_path, repo_name in repo_results:
        if not repo_path: continue
        if repo_name not in multi_repo_data: multi_repo_data[repo_name] = {}
            
        for root, _, files in os.walk(repo_path):
            if ".git" in root: continue
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)
                if is_valid_file(file):
                    read_tasks.append((async_read_file(full_path, rel_path), repo_name, rel_path))

    print(f"\n📖 Reading files across {len(multi_repo_data)} repositories...")
    file_contents = await asyncio.gather(*[t[0] for t in read_tasks])
    
    total_files = 0
    for i, content in enumerate(file_contents):
        _, r_name, r_path = read_tasks[i]
        if content and content.strip():
            multi_repo_data[r_name][r_path] = {"content": content}
            total_files += 1

    print(f"✅ Total Loaded: {total_files} files.")
    return multi_repo_data

# --- 2. Advanced Parsers (Tree-sitter) ---

# We define Tree-sitter queries to extract Definitions (functions/classes) and Calls.
QUERIES = {
    'python': {
        'defs': """
            (function_definition name: (identifier) @name) @def
            (class_definition name: (identifier) @name) @def
        """,
        'calls': """(call function: (identifier) @call)"""
    },
    'javascript': {
        'defs': """
            (function_declaration name: (identifier) @name) @def
            (class_declaration name: (identifier) @name) @def
            (method_definition name: (property_identifier) @name) @def
            (variable_declarator 
                name: (identifier) @name 
                value: [(arrow_function) (function)]) @def
        """,
        'calls': """(call_expression function: (identifier) @call)"""
    },
    'typescript': {
        'defs': """
            (function_declaration name: (identifier) @name) @def
            (class_declaration name: (type_identifier) @name) @def
            (method_definition name: (property_identifier) @name) @def
            (variable_declarator 
                name: (identifier) @name 
                value: [(arrow_function) (function)]) @def
        """,
        'calls': """(call_expression function: (identifier) @call)"""
    },
    'java': {
        'defs': """
            (method_declaration name: (identifier) @name) @def
            (class_declaration name: (identifier) @name) @def
        """,
        'calls': """(method_invocation name: (identifier) @call)"""
    },
    'cpp': {
        'defs': """
            (function_definition declarator: (function_declarator declarator: (identifier) @name)) @def
            (class_specifier name: (type_identifier) @name) @def
        """,
        'calls': """(call_expression function: (identifier) @call)"""
    },
    'css': {
        'defs': """
            (rule_set (selectors) @name) @def
            (media_statement) @def
        """,
        'calls': "" # CSS generally doesn't "call" things in the same way
    },
    'bash': {
        'defs': """
            (function_definition name: (word) @name) @def
        """,
        'calls': """(command_name (word) @call)"""
    }
}

def get_node_text(node, content_bytes):
    return content_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')

def parse_with_treesitter(content, language_name):
    """
    Robust parser that handles recent Tree-sitter API changes (v0.22+).
    """
    if language_name not in QUERIES:
        return None

    try:
        # 1. Get Language Object (using helper)
        language = get_language(language_name)
        
        # 2. Instantiate Parser (Manually to avoid wrapper errors)
        parser = Parser()
        # Handle API differences for setting language
        if hasattr(parser, 'language'):
            parser.language = language  # New API (v0.22+)
        else:
            parser.set_language(language) # Old API
        
        # 3. Parse content
        content_bytes = bytes(content, "utf8")
        tree = parser.parse(content_bytes)
        
        # 4. Prepare Queries (using explicit Query constructor)
        lang_queries = QUERIES[language_name]
        
        try:
            def_query = Query(language, lang_queries['defs'])
        except TypeError:
            # Fallback for very old versions (unlikely but safe)
            def_query = language.query(lang_queries['defs'])

        call_query = None
        if lang_queries['calls']:
            try:
                call_query = Query(language, lang_queries['calls'])
            except TypeError:
                call_query = language.query(lang_queries['calls'])

        results = []

        # 5. Execute Query
        # .captures() returns different formats in different versions.
        # v0.24+: [(Node, str)]
        # Older:  [(Node, str)] or similar
        captures = def_query.captures(tree.root_node)
        
        # Map definition nodes to their names
        def_map = {} # {id(node): {'name': str, 'node': node}}

        for capture in captures:
            # Handle tuple unpacking safely
            if isinstance(capture, tuple):
                node, tag = capture
            else:
                # Some versions might return objects
                node = capture.node
                tag = capture.name

            if tag == 'def':
                if id(node) not in def_map:
                    def_map[id(node)] = {'node': node, 'name': 'unknown'}
            elif tag == 'name':
                # Associate name with its parent definition
                parent = node.parent
                while parent:
                    if id(parent) in def_map:
                        def_map[id(parent)]['name'] = get_node_text(node, content_bytes)
                        break
                    parent = parent.parent

        # 6. Extract Calls & Build Node Objects
        for def_id, info in def_map.items():
            def_node = info['node']
            name = info['name']
            
            code_text = get_node_text(def_node, content_bytes)
            start_line = def_node.start_point[0] + 1
            
            calls = set()
            if call_query:
                call_captures = call_query.captures(def_node)
                for capture in call_captures:
                    if isinstance(capture, tuple):
                        call_node, _ = capture
                    else:
                        call_node = capture.node
                    
                    call_name = get_node_text(call_node, content_bytes)
                    calls.add(call_name)

            results.append({
                "name": name,
                "type": "definition",
                "code": code_text,
                "line_start": start_line,
                "calls": list(calls)
            })
            
        return {"nodes": results, "imports": [], "globals": ""}

    except Exception as e:
        print(f"   ⚠️ Tree-sitter error for {language_name}: {e}")
        # Return empty result to allow pipeline to continue with other files
        return {"nodes": [], "imports": [], "globals": ""}

# --- 3. Parsing Router ---

async def parse_file(filename, content):
    """Routes file to correct parser."""
    
    # Python (Keep AST for simplicity as it's built-in and perfect)
    if filename.endswith(".py"):
        from ast import parse
        # Reuse previous AST logic for Python to avoid regression
        # (Included inline for completeness)
        try:
            tree = ast.parse(content)
            visitor = PythonFunctionVisitor(content)
            visitor.visit(tree)
            return {
                "nodes": visitor.nodes, 
                "imports": list(visitor.imports),
                "globals": "\n".join(visitor.global_context)
            }
        except: return None

    # Tree-sitter Languages
    lang_map = {
        '.js': 'javascript', '.jsx': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp', '.cc': 'cpp', '.c': 'cpp', '.h': 'cpp',
        '.css': 'css', '.scss': 'css',
        '.sh': 'bash'
    }
    
    ext = os.path.splitext(filename)[1].lower()
    if ext in lang_map:
        return parse_with_treesitter(content, lang_map[ext])
        
    return None

# --- Python AST Helpers (Kept for Python Support) ---
class PythonFunctionVisitor(ast.NodeVisitor):
    def __init__(self, content):
        self.content = content
        self.nodes = [] 
        self.imports = set()
        self.global_context = [] 
    def get_code(self, node): return ast.get_source_segment(self.content, node)
    def visit_Import(self, node):
        self.global_context.append(self.get_code(node))
        for alias in node.names: self.imports.add(alias.name)
    def visit_ImportFrom(self, node):
        self.global_context.append(self.get_code(node))
        if node.module: self.imports.add(node.module)
        for alias in node.names: self.imports.add(alias.name)
    def visit_FunctionDef(self, node): self._handle(node, "function")
    def visit_AsyncFunctionDef(self, node): self._handle(node, "function")
    def visit_ClassDef(self, node): self._handle(node, "class")
    def _handle(self, node, node_type):
        call_visitor = CallExtractor()
        call_visitor.visit(node)
        self.nodes.append({
            "name": node.name, "type": node_type,
            "code": self.get_code(node), "line_start": node.lineno,
            "calls": list(call_visitor.calls)
        })

class CallExtractor(ast.NodeVisitor):
    def __init__(self): self.calls = set()
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name): self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute): self.calls.add(node.func.attr)
        self.generic_visit(node)

# --- 4. Enhanced Graph Builder ---

async def build_single_repo_graph(repo_name, files_data):
    symbol_registry = {} 
    file_metadata = {} 

    print(f"   🔨 Parsing {repo_name} using Tree-sitter...")

    # 1. Parse Definitions
    for filename, data in files_data.items():
        content = data['content']
        parsed = await parse_file(filename, content)
            
        if parsed:
            file_metadata[filename] = {
                "imports": set(parsed['imports']),
                "globals": parsed['globals']
            }
            for node in parsed['nodes']:
                name = node['name']
                if name not in symbol_registry: symbol_registry[name] = []
                symbol_registry[name].append({
                    "file": filename, "type": node['type'], 
                    "code": node['code'], "line": node['line_start'], "calls": node['calls']
                })
    
    # 2. Link Edges
    graph = {} 
    defined_symbols = set(symbol_registry.keys())
    node_indegree = Counter()

    for sym_name, implementations in symbol_registry.items():
        for impl in implementations:
            node_id = f"{impl['file']}::{sym_name}"
            # Safe access to metadata (CSS might not have imports)
            meta = file_metadata.get(impl['file'], {"imports": [], "globals": ""})
            current_file_imports = meta['imports']
            
            valid_deps = []
            for called in impl['calls']:
                if called in defined_symbols:
                    targets = symbol_registry[called]
                    best_match = None
                    
                    # PRIORITY 1: Same File
                    for t in targets:
                        if t['file'] == impl['file']:
                            best_match = t
                            break
                    
                    # PRIORITY 2: Imported File 
                    if not best_match:
                        for t in targets:
                            t_fname = os.path.splitext(os.path.basename(t['file']))[0]
                            if t_fname in current_file_imports:
                                best_match = t
                                break
                    
                    # PRIORITY 3: Global Fallback
                    if not best_match and len(targets) == 1:
                         best_match = targets[0]

                    if best_match:
                        tid = f"{best_match['file']}::{called}"
                        if tid != node_id:
                            valid_deps.append(tid)
                            node_indegree[tid] += 1
            
            graph[node_id] = {
                "file": impl['file'],
                "code": impl['code'],
                "type": impl['type'],
                "line": impl['line'],
                "globals": meta['globals'],
                "dependencies": list(set(valid_deps))
            }
    
    for node_id, data in graph.items():
        data['is_super_node'] = node_indegree[node_id] > UBIQUITOUS_THRESHOLD

    print(f"   ✅ Graph for [{repo_name}]: {len(graph)} nodes")
    return graph

async def build_multi_symbol_graph(multi_repo_data):
    multi_graph = {}
    for repo_name, files_data in multi_repo_data.items():
        multi_graph[repo_name] = await build_single_repo_graph(repo_name, files_data)
    with open(KNOWLEDGE_BASE_FILE, 'w') as f: json.dump(multi_graph, f, indent=2)
    return multi_graph

# --- 5. Enhanced Agents ---

async def reframer_agent(user_query, chat_history, available_repos):
    print("🧠 Reframer: Detecting Target Repo...", flush=True)
    history_text = ""
    for turn in chat_history[-3:]: history_text += f"{turn['role'].upper()}: {turn['content']}\n"

    prompt = f"""
    Repos: [{", ".join(available_repos)}]
    History: {history_text}
    Query: "{user_query}"
    
    Task: Identify TARGET_REPO (or 'ALL') and rewrite the QUERY technically.
    Format:
    TARGET_REPO: <name>
    QUERY: <text>
    """
    res = await safe_chat_completion(MODEL_NAME, [{"role": "user", "content": prompt}])
    content = res.choices[0].message.content.strip()
    
    target = "ALL"
    query = user_query
    
    m_repo = re.search(r"TARGET_REPO:\s*(.+)", content)
    if m_repo: target = m_repo.group(1).strip()
    m_query = re.search(r"QUERY:\s*(.+)", content, re.DOTALL)
    if m_query: query = m_query.group(1).strip()
    
    return target, query

def rank_nodes_heuristic(query, nodes_dict):
    query_terms = set(re.findall(r'\w+', query.lower()))
    ranked = []
    
    for nid, data in nodes_dict.items():
        score = 0
        name_parts = nid.lower().split('::')[-1]
        
        # Exact match
        if name_parts == query.lower(): score += 50
        # Partial match
        if name_parts in query.lower(): score += 10
        if query.lower() in name_parts: score += 10
        
        for term in query_terms:
            if term in name_parts: score += 5
        
        if score > 0:
            ranked.append((score, nid, data['type']))
            
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [f"ID: {r[1]} | Type: {r[2]}" for r in ranked[:500]]

async def selector_agent(target_repo, technical_query, multi_graph):
    print(f"🗂️  Selector: Smart Ranking & Pruning...", flush=True)
    
    active_graphs = {}
    if target_repo == "ALL" or target_repo not in multi_graph:
        for r_name, g_data in multi_graph.items():
            for nid, ndata in g_data.items(): active_graphs[f"{r_name}::{nid}"] = ndata
    else:
        active_graphs = multi_graph[target_repo]
    
    if not active_graphs: return []

    ranked_menu = rank_nodes_heuristic(technical_query, active_graphs)
    
    if not ranked_menu:
        keys = list(active_graphs.keys())[:500]
        ranked_menu = [f"ID: {k} | Type: {active_graphs[k]['type']}" for k in keys]
    
    menu_str = "\n".join(ranked_menu)

    prompt = f"""
    Query: "{technical_query}"
    Candidate Nodes (Ranked by relevance):
    {menu_str}
    
    Select 5-10 STARTING seed nodes.
    Return JSON: {{ "seed_nodes": ["id1", "id2", ...] }}
    """
    
    res = await safe_chat_completion(MODEL_NAME, [{"role": "user", "content": prompt}], {"type": "json_object"})
    seeds = json.loads(res.choices[0].message.content).get('seed_nodes', [])
    
    selected = set()
    queue = [s for s in seeds if s in active_graphs]
    selected.update(queue)
    
    depth = 0
    while queue and depth < MAX_RECURSION_DEPTH:
        next_q = []
        for curr in queue:
            node = active_graphs[curr]
            
            if node.get('is_super_node', False): continue

            for dep in node['dependencies']:
                actual_id = dep
                if target_repo == "ALL" and "::" in curr:
                    prefix = curr.split("::")[0]
                    if not dep.startswith(prefix): actual_id = f"{prefix}::{dep}"
                
                if actual_id in active_graphs and actual_id not in selected:
                    selected.add(actual_id)
                    next_q.append(actual_id)
        queue = next_q
        depth += 1

    context = []
    for nid in selected:
        node = active_graphs[nid]
        code_lines = node['code'].splitlines()
        numbered_code = []
        for i, line in enumerate(code_lines):
            numbered_code.append(f"{node['line'] + i}: {line}")
        
        final_code = "\n".join(numbered_code)
        context.append(f"=== {node['file']} (Lines {node['line']}-...) ===\n{node['globals']}\n...\n{final_code}")
    
    print(f"   🕸️  Selected {len(selected)} nodes.")
    return context

async def answering_agent(user_query, context):
    print("📝 Answering Agent...", flush=True)
    res = await safe_chat_completion(
        MODEL_NAME, 
        [{"role": "user", "content": f"Query: {user_query}\n\nCode Context (With Line Numbers):\n" + "\n".join(context)}]
    )
    return res.choices[0].message.content

# --- Main ---

async def main():
    chat_history = []
    try:
        print("🔗 === PRO MULTI-LANG CODE NAVIGATOR (Tree-sitter) === 🔗")
        gh_input = input("\n🐙 Enter GitHub Repos (comma-separated): ")
        if not gh_input.strip(): return

        multi_repo_data = await ingest_sources(gh_input)
        if not multi_repo_data: return

        multi_graph = await build_multi_symbol_graph(multi_repo_data)
        available_repos = list(multi_graph.keys())
        
        while True:
            query = input("\n" + "-"*40 + "\n(Type 'exit' to quit) Question: ")
            if query.lower() in ['exit', 'quit']: break
            
            target, re_query = await reframer_agent(query, chat_history, available_repos)
            context = await selector_agent(target, re_query, multi_graph)
            
            if not context:
                print("   ⚠️ No relevant code found.")
                continue

            ans = await answering_agent(query, context)
            print("\n" + "="*60 + f"\n✅ ANSWER:\n{ans}\n" + "="*60)
            
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": ans})

    except KeyboardInterrupt: print("\n⚠️ Interrupted.")
    except Exception as e: print(f"\n❌ Error: {e}")
    finally: perform_cleanup()

if __name__ == "__main__":
    asyncio.run(main())