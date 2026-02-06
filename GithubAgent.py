import os
import json
import asyncio
import shutil
import stat
import errno
import sys
import re
import ast
import hashlib
from typing import List, Dict, Optional, Set, Any
from git import Repo
from openai import AsyncOpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"

# Graph Configuration
MAX_RECURSION_DEPTH = 3
MAX_CONCURRENCY = 50 
MAX_CONTEXT_CHARS = 200000 

# Cache Configuration
CACHE_FILE = "symbol_graph_cache.json"

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not found in .env. Please set it.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Paths
TEMP_DIR = "./temp_session_data"
KNOWLEDGE_BASE_FILE = "symbol_graph.json"

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
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.h', '.cs', '.go', '.rs', '.rb', '.php'
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
    
    # Extract clean repo name for isolation
    # e.g., https://github.com/user/my-repo.git -> my-repo
    clean_name = url.split("/")[-1].replace(".git", "")
    if not clean_name: clean_name = f"repo_{source_id}"
    
    # Ensure unique names if user inputs duplicates
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
    
    dir_tasks = []
    for i, url in enumerate(git_urls):
        dir_tasks.append(handle_github_repo(url, i))
        
    # results is list of (path, name) tuples
    repo_results = await asyncio.gather(*dir_tasks)
    
    # Structure: { "repo_name": { "filepath": { "content": ... } } }
    multi_repo_data = {}
    
    read_tasks = []
    
    for repo_path, repo_name in repo_results:
        if not repo_path: continue
        
        if repo_name not in multi_repo_data:
            multi_repo_data[repo_name] = {}
            
        for root, _, files in os.walk(repo_path):
            if ".git" in root: continue
            for file in files:
                full_path = os.path.join(root, file)
                # Keep path relative to the specific repo root
                rel_path = os.path.relpath(full_path, repo_path)
                
                if is_valid_file(file):
                    # We store a tuple reference to update the dict later
                    read_tasks.append((async_read_file(full_path, rel_path), repo_name, rel_path))

    print(f"\n📖 Reading files across {len(multi_repo_data)} repositories...")
    
    # Execute reads
    file_contents = await asyncio.gather(*[t[0] for t in read_tasks])
    
    total_files = 0
    for i, content in enumerate(file_contents):
        _, r_name, r_path = read_tasks[i]
        if content and content.strip():
            multi_repo_data[r_name][r_path] = {"content": content}
            total_files += 1

    print(f"✅ Total Loaded: {total_files} files across {list(multi_repo_data.keys())}.")
    return multi_repo_data

# --- 2. AST Symbol Parser ---

class PythonFunctionVisitor(ast.NodeVisitor):
    def __init__(self, content):
        self.content = content
        self.nodes = [] 
        self.global_context = [] 
        self.current_scope = None

    def get_code(self, node):
        return ast.get_source_segment(self.content, node)

    def visit_Import(self, node):
        self.global_context.append(self.get_code(node))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.global_context.append(self.get_code(node))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._handle_func_or_class(node, "function")

    def visit_AsyncFunctionDef(self, node):
        self._handle_func_or_class(node, "function")

    def visit_ClassDef(self, node):
        self._handle_func_or_class(node, "class")

    def _handle_func_or_class(self, node, node_type):
        name = node.name
        code = self.get_code(node)
        call_visitor = CallExtractor()
        call_visitor.visit(node)
        
        self.nodes.append({
            "name": name,
            "type": node_type,
            "code": code,
            "calls": list(call_visitor.calls)
        })

class CallExtractor(ast.NodeVisitor):
    def __init__(self):
        self.calls = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.add(node.func.attr)
        self.generic_visit(node)

def parse_python_symbols(filename, content):
    try:
        tree = ast.parse(content)
        visitor = PythonFunctionVisitor(content)
        visitor.visit(tree)
        return {
            "nodes": visitor.nodes,
            "globals": "\n".join(visitor.global_context)
        }
    except:
        return {"nodes": [], "globals": ""}

async def build_single_repo_graph(repo_name, files_data):
    """Builds a graph for a single repository, isolated from others."""
    symbol_registry = {} 
    file_globals = {} 

    # 1. Parse Definitions
    for filename, data in files_data.items():
        if filename.endswith(".py"):
            result = parse_python_symbols(filename, data['content'])
            file_globals[filename] = result['globals']
            
            for node in result['nodes']:
                name = node['name']
                if name not in symbol_registry:
                    symbol_registry[name] = []
                
                symbol_registry[name].append({
                    "file": filename,
                    "type": node['type'],
                    "code": node['code'],
                    "calls": node['calls']
                })
    
    # 2. Link Edges
    graph = {} 
    defined_symbols = set(symbol_registry.keys())

    for sym_name, implementations in symbol_registry.items():
        for impl in implementations:
            node_id = f"{impl['file']}::{sym_name}"
            
            valid_deps = []
            for called_func in impl['calls']:
                if called_func in defined_symbols:
                    targets = symbol_registry[called_func]
                    for target in targets:
                        target_id = f"{target['file']}::{called_func}"
                        if target_id != node_id:
                            valid_deps.append(target_id)
            
            graph[node_id] = {
                "file": impl['file'],
                "code": impl['code'],
                "type": impl['type'],
                "globals": file_globals.get(impl['file'], ""),
                "dependencies": list(set(valid_deps))
            }
            
    print(f"   Built Graph for [{repo_name}]: {len(graph)} nodes")
    return graph

async def build_multi_symbol_graph(multi_repo_data):
    print("\n🕵️  Graph Builder: Building isolated graphs per repo...")
    
    multi_graph = {}
    
    for repo_name, files_data in multi_repo_data.items():
        multi_graph[repo_name] = await build_single_repo_graph(repo_name, files_data)
        
    with open(KNOWLEDGE_BASE_FILE, 'w') as f: json.dump(multi_graph, f, indent=2)
    return multi_graph

# --- 3. Context-Aware Reframer ---

async def reframer_agent(user_query, chat_history, available_repos):
    print("🧠 Reframer: Detecting Target Repo...", flush=True)
    history_text = ""
    for turn in chat_history[-3:]: history_text += f"{turn['role'].upper()}: {turn['content']}\n"

    repo_list_str = ", ".join(available_repos)

    prompt = f"""
    You are a Technical Assistant managing multiple repositories.
    Available Repositories: [{repo_list_str}]
    
    Conversation History:
    {history_text}
    
    Current Query: "{user_query}"
    
    TASK:
    1. Determine which Repository the user is referring to. 
       - If they mention a specific repo name, use that.
       - If context implies one (e.g., following up on previous questions), use that.
       - If it's ambiguous or applies to all, use 'ALL'.
    2. Rewrite the query to be a precise technical search.
    
    OUTPUT FORMAT:
    TARGET_REPO: <repo_name_or_ALL>
    QUERY: <rewritten_query>
    """
    
    res = await safe_chat_completion(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    content = res.choices[0].message.content.strip()
    
    # Parse output
    target_repo = "ALL"
    rewritten_query = user_query
    
    match_repo = re.search(r"TARGET_REPO:\s*(.+)", content)
    if match_repo: target_repo = match_repo.group(1).strip()
    
    match_query = re.search(r"QUERY:\s*(.+)", content, re.DOTALL)
    if match_query: rewritten_query = match_query.group(1).strip()
    
    print(f"   ↳ Target: [{target_repo}] | Query: \"{rewritten_query}\"")
    return target_repo, rewritten_query

# --- 4. Symbol Graph Selector ---

async def selector_agent(target_repo, technical_query, multi_graph):
    print(f"🗂️  Selector: Picking Function Nodes in [{target_repo}]...", flush=True)
    
    # 1. Determine active graph(s)
    active_graphs = {}
    
    if target_repo == "ALL" or target_repo not in multi_graph:
        # Flatten all graphs (prefix IDs to avoid collision)
        for r_name, g_data in multi_graph.items():
            for node_id, node_data in g_data.items():
                # Store with unique key: repo::node_id
                active_graphs[f"{r_name}::{node_id}"] = node_data
    else:
        # Use specific repo graph
        active_graphs = multi_graph[target_repo]
    
    if not active_graphs:
        return []

    # 2. Create Menu
    menu = []
    # Limit menu size randomly if too big, or use smart sampling
    keys = list(active_graphs.keys())
    if len(keys) > 800: keys = keys[:800]
    
    for node_id in keys:
        data = active_graphs[node_id]
        menu.append(f"ID: {node_id} | Type: {data['type']}")
    
    menu_str = "\n".join(menu)

    prompt = f"""
    You are a Code Navigator.
    Query: "{technical_query}"
    Context Repo: {target_repo}
    
    Available Nodes:
    {menu_str}
    
    TASK:
    Select 2-4 STARTING NODES (IDs) most relevant to the query.
    Return JSON: {{ "seed_nodes": ["node_id_1", "node_id_2"] }}
    """
    
    res = await safe_chat_completion(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    seed_nodes = json.loads(res.choices[0].message.content).get('seed_nodes', [])
    print(f"   🌱 Seeds: {seed_nodes}")

    # 3. Traversal
    selected_nodes = set()
    queue = []
    
    for seed in seed_nodes:
        if seed in active_graphs:
            queue.append(seed)
            selected_nodes.add(seed)

    current_depth = 0
    while queue and current_depth < MAX_RECURSION_DEPTH:
        next_queue = []
        for current_node_id in queue:
            node = active_graphs[current_node_id]
            deps = node['dependencies']
            
            for dep_id in deps:
                # Handle ID prefixing if we are in ALL mode
                if dep_id not in active_graphs and target_repo == "ALL":
                     # Try to find prefixed version? 
                     # Actually, dependencies inside the node data are already relative to their own graph.
                     # In ALL mode, we prefixed the KEYS, but the dependency strings inside the values are original.
                     # We need to re-prefix them to find them in active_graphs.
                     # This complexity implies ALL mode is tricky.
                     # Simplified: If ALL, we blindly assume unique filenames or fuzzy match.
                     # Better: Let's stick to strict matching. If not found, skip.
                     pass
                
                if dep_id in active_graphs and dep_id not in selected_nodes:
                    selected_nodes.add(dep_id)
                    next_queue.append(dep_id)
                elif target_repo == "ALL":
                    # Try finding the prefixed key
                    # Current node key is "repo::file::func". 
                    # Dependency is "file2::func2".
                    # We need "repo::file2::func2".
                    repo_prefix = current_node_id.split("::")[0]
                    prefixed_dep = f"{repo_prefix}::{dep_id}"
                    if prefixed_dep in active_graphs and prefixed_dep not in selected_nodes:
                        selected_nodes.add(prefixed_dep)
                        next_queue.append(prefixed_dep)
        
        queue = next_queue
        current_depth += 1

    # 4. Construct Output
    files_context = {} 
    
    for node_id in selected_nodes:
        node = active_graphs[node_id]
        
        # In ALL mode, we might want to display Repo Name too
        fname = node['file']
        if target_repo == "ALL":
             # Extract repo from key if needed, or just append to filename
             repo_prefix = node_id.split("::")[0] if "::" in node_id else "UNKNOWN"
             display_name = f"[{repo_prefix}] {fname}"
        else:
             display_name = fname
        
        if display_name not in files_context:
            files_context[display_name] = {
                "globals": node['globals'],
                "functions": []
            }
        files_context[display_name]["functions"].append(node['code'])

    final_output = []
    for name, data in files_context.items():
        block = f"=== FILE: {name} ===\n"
        block += f"{data['globals']}\n"
        block += "\n# ... (hidden) ...\n\n"
        block += "\n\n".join(data['functions'])
        final_output.append(block)

    print(f"   🕸️  Selected {len(selected_nodes)} functions.")
    return final_output

# --- 5. Answering Agent ---

async def answering_agent(user_query, context_strings):
    print("📝 Answering Agent: Generating response...", flush=True)
    full_context = "\n".join(context_strings)
    res = await safe_chat_completion(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a senior developer. Answer based strictly on the provided Code Context."},
            {"role": "user", "content": f"Query: {user_query}\n\nCode Context:\n{full_context}"}
        ]
    )
    return res.choices[0].message.content

# --- Main Orchestrator ---

async def main():
    chat_history = []
    try:
        print("🔗 === GITHUB SOURCE CONFIGURATION === 🔗")
        gh_input = input("\n🐙 Enter GitHub Repos (comma-separated): ")
        if not gh_input.strip(): return

        # Load separate repos
        multi_repo_data = await ingest_sources(gh_input)
        if not multi_repo_data: return

        # Build isolated graphs
        multi_graph = await build_multi_symbol_graph(multi_repo_data)
        
        available_repos = list(multi_graph.keys())
        
        while True:
            query = input("\n" + "-"*40 + "\n(Type 'exit' to quit) Question: ")
            if query.lower() in ['exit', 'quit']: break
            
            # Reframe with Repo Awareness
            target_repo, technical_query = await reframer_agent(query, chat_history, available_repos)
            
            # Select from Specific Graph
            context_strings = await selector_agent(target_repo, technical_query, multi_graph)
            
            if not context_strings:
                print("   ⚠️ No relevant code found.")
                continue

            # Answer
            answer = await answering_agent(query, context_strings)
            print("\n" + "="*60 + f"\n✅ ANSWER:\n{answer}\n" + "="*60)
            
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})

    except KeyboardInterrupt: print("\n⚠️ Interrupted.")
    except Exception as e: print(f"\n❌ Error: {e}")
    finally: perform_cleanup()

if __name__ == "__main__":
    asyncio.run(main())