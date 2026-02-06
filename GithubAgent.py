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
MAX_RECURSION_DEPTH = 3   # We can go deeper now because nodes are smaller
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
    # For Function Graph, we prioritize code files.
    name = os.path.basename(filename).lower()
    ext = os.path.splitext(filename)[1].lower()
    return name in ALWAYS_KEEP_NAMES or ext in ALLOWED_EXTENSIONS

def read_universal_text(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f: return f.read()
    except: return ""

async def async_read_file(path, relative_path):
    # We strictly only care about text/code for AST parsing
    return await asyncio.to_thread(read_universal_text, path)

async def handle_github_repo(url, source_id):
    if not url: return None
    repo_path = os.path.join(TEMP_DIR, f"repo_{source_id}")
    print(f"🔄 Cloning GitHub Repo: {url}...")
    try:
        await asyncio.to_thread(Repo.clone_from, url, repo_path)
        return repo_path
    except Exception as e:
        print(f"❌ Git Clone Failed: {e}")
        return None

async def ingest_sources(github_inputs: str):
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR, onerror=handle_remove_readonly)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    tasks = []
    paths = []
    
    git_urls = [s.strip() for s in github_inputs.split(',') if s.strip()]
    
    dir_tasks = []
    for i, url in enumerate(git_urls):
        dir_tasks.append(handle_github_repo(url, i))
        
    local_dirs = await asyncio.gather(*dir_tasks)
    
    for local_dir in local_dirs:
        if not local_dir: continue
        for root, _, files in os.walk(local_dir):
            if ".git" in root: continue
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, TEMP_DIR)
                if is_valid_file(file):
                    tasks.append(async_read_file(full_path, rel_path))
                    paths.append(rel_path)

    print(f"\n📖 Processing gathered files...")
    results = await asyncio.gather(*tasks)
    
    files_data = {}
    for p, c in zip(paths, results):
        if c and c.strip(): 
            files_data[p] = {"content": c}

    print(f"\n✅ Total Loaded: {len(files_data)} items.")
    return files_data

# --- 2. AST Symbol Parser (The Engine) ---

class PythonFunctionVisitor(ast.NodeVisitor):
    def __init__(self, content):
        self.content = content
        self.nodes = [] # List of dicts: {name, type, code, calls}
        self.global_context = [] # Imports and global assignments
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
        # We only track top-level or class-level (not nested funcs for now to keep graph clean)
        # unless user wants extreme detail. Let's stick to top-level/class-level.
        
        name = node.name
        code = self.get_code(node)
        
        # Extract calls made INSIDE this node
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
        # Extract function name from calls like func() or module.func()
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
    except Exception as e:
        # print(f"AST Error in {filename}: {e}")
        return {"nodes": [], "globals": ""}

async def build_symbol_graph(files_data):
    print("\n🕵️  Graph Builder: Parsing Symbols & Linking Nodes...")
    
    # 1. Parse all files to find Definitions (Nodes)
    # Registry: { "function_name": [ {file, code, calls} ] }
    # List because multiple files might have 'main' or 'utils'
    symbol_registry = {} 
    file_globals = {} # filename -> string of imports/globals

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
                    "calls": node['calls'] # These are candidate calls
                })
        else:
            # Non-python files: Treat whole file as one node for now (Fallback)
            # Or use LLM to extract symbols if needed. 
            pass

    # 2. Link Edges (Resolve Calls)
    # We only link if the called function exists in our Registry (Internal Dependency)
    # This automatically filters out numpy, pandas, etc.
    
    graph = {} # "filename::symbolname" -> { metadata, dependencies: [] }
    
    defined_symbols = set(symbol_registry.keys())

    for sym_name, implementations in symbol_registry.items():
        for impl in implementations:
            # Create a unique ID for the node
            node_id = f"{impl['file']}::{sym_name}"
            
            # Filter calls: Only keep calls that exist in our repo
            valid_deps = []
            for called_func in impl['calls']:
                if called_func in defined_symbols:
                    # Logic to resolve WHICH file it comes from.
                    # Simple heuristic: If multiple files have 'util_func', we link to all (ambiguous) 
                    # or prefer same directory. For now, link all matches.
                    targets = symbol_registry[called_func]
                    for target in targets:
                        target_id = f"{target['file']}::{called_func}"
                        # Don't self-reference
                        if target_id != node_id:
                            valid_deps.append(target_id)
            
            graph[node_id] = {
                "file": impl['file'],
                "code": impl['code'],
                "type": impl['type'],
                "globals": file_globals.get(impl['file'], ""),
                "dependencies": list(set(valid_deps)) # Dedup
            }

    print(f"   ✅ Graph Built: {len(graph)} function/class nodes created.")
    
    # Save for debugging
    with open(KNOWLEDGE_BASE_FILE, 'w') as f: json.dump(graph, f, indent=2)
    return graph

# --- 3. Reframer Agent ---
async def reframer_agent(user_query, chat_history):
    print("🧠 Reframer: Clarifying intent...", flush=True)
    history_text = ""
    for turn in chat_history[-3:]: history_text += f"{turn['role'].upper()}: {turn['content']}\n"

    prompt = f"""
    Rewrite the user's query into a precise technical search query.
    History: {history_text}
    Query: "{user_query}"
    Return ONLY the rewritten query.
    """
    res = await safe_chat_completion(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    new_query = res.choices[0].message.content.strip()
    print(f"   ↳ Resolved: \"{new_query}\"")
    return new_query

# --- 4. Symbol Graph Selector ---

async def selector_agent(technical_query, graph):
    print(f"🗂️  Selector: Picking Function Nodes...", flush=True)
    
    # 1. Create a "Menu" for the LLM
    # We can't dump code. Just signatures.
    menu = []
    for node_id, data in graph.items():
        menu.append(f"ID: {node_id} | Type: {data['type']}")
    
    # If menu is too huge, we might need embeddings. 
    # For now, let's assume < 500 functions or truncate.
    menu_str = "\n".join(menu[:800]) # Hard limit for safety

    prompt = f"""
    You are a Code Navigator.
    Query: "{technical_query}"
    
    Available Functions/Classes (Nodes):
    {menu_str}
    
    TASK:
    Select 2-4 STARTING NODES (IDs) that are most likely to handle the logic for the query.
    Do not pick utility functions unless core to the query.
    
    Return JSON: {{ "seed_nodes": ["file.py::func_name"] }}
    """
    
    res = await safe_chat_completion(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    seed_nodes = json.loads(res.choices[0].message.content).get('seed_nodes', [])
    print(f"   🌱 Seeds: {seed_nodes}")

    # 2. Graph Traversal (DFS/BFS)
    selected_nodes = set()
    queue = []
    
    # Validate seeds
    for seed in seed_nodes:
        if seed in graph:
            queue.append(seed)
            selected_nodes.add(seed)

    current_depth = 0
    while queue and current_depth < MAX_RECURSION_DEPTH:
        next_queue = []
        for current_node_id in queue:
            node = graph[current_node_id]
            deps = node['dependencies']
            
            for dep_id in deps:
                if dep_id not in selected_nodes:
                    selected_nodes.add(dep_id)
                    next_queue.append(dep_id)
        
        queue = next_queue
        current_depth += 1

    # 3. Construct Context
    # We group by FILE to avoid repeating imports/globals
    
    files_context = {} # filename -> list of function codes
    
    for node_id in selected_nodes:
        node = graph[node_id]
        fname = node['file']
        
        if fname not in files_context:
            files_context[fname] = {
                "globals": node['globals'],
                "functions": []
            }
        
        files_context[fname]["functions"].append(node['code'])

    # Final Text Construction
    final_output = []
    for fname, data in files_context.items():
        block = f"=== FILE: {fname} ===\n"
        block += f"{data['globals']}\n" # Imports
        block += "\n# ... (unrelated code hidden) ...\n\n"
        block += "\n\n".join(data['functions'])
        final_output.append(block)

    print(f"   🕸️  Selected {len(selected_nodes)} functions across {len(files_context)} files.")
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

        files_data = await ingest_sources(gh_input)
        if not files_data: return

        # 1. Build Function Graph (AST)
        graph = await build_symbol_graph(files_data)
        
        while True:
            query = input("\n" + "-"*40 + "\n(Type 'exit' to quit) Question: ")
            if query.lower() in ['exit', 'quit']: break
            
            # 2. Reframe
            technical_query = await reframer_agent(query, chat_history)
            
            # 3. Select (Function Graph Traversal)
            context_strings = await selector_agent(technical_query, graph)
            
            if not context_strings:
                print("   ⚠️ No relevant code found.")
                continue

            # 4. Answer
            answer = await answering_agent(query, context_strings)
            print("\n" + "="*60 + f"\n✅ ANSWER:\n{answer}\n" + "="*60)
            
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})

    except KeyboardInterrupt: print("\n⚠️ Interrupted.")
    except Exception as e: print(f"\n❌ Error: {e}")
    finally: perform_cleanup()

if __name__ == "__main__":
    asyncio.run(main())