import os
import json
import asyncio
import shutil
import stat
import errno
import sys
import re
from typing import List, Dict, Optional, Set, Union
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
MAX_RECURSION_DEPTH = 5   # Trace dependencies 5 levels deep
MAX_CONCURRENCY = 50 
MAX_CONTEXT_CHARS = 200000 

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not found in .env. Please set it.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Paths
TEMP_DIR = "./temp_session_data"
SUMMARIES_DIR = "./repo_summaries"
KNOWLEDGE_BASE_FILE = "session_knowledge.json"

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
    if os.path.exists(SUMMARIES_DIR):
        try: shutil.rmtree(SUMMARIES_DIR, onerror=handle_remove_readonly)
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
        '.md', '.markdown', '.txt', '.pdf', '.docx', '.rst',
        '.py', '.ipynb', '.js', '.jsx', '.ts', '.tsx', '.vue', '.svelte', 
        '.html', '.css', '.scss', '.java', '.kt', '.scala', '.c', '.cpp', 
        '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.sh', '.bat', 
        '.ps1', '.swift', '.sql', '.json', '.yaml', '.yml', '.xml', '.toml'
    }
    name = os.path.basename(filename).lower()
    ext = os.path.splitext(filename)[1].lower()
    return name in ALWAYS_KEEP_NAMES or ext in ALLOWED_EXTENSIONS

def read_universal_text(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f: return f.read()
    except: return ""

def read_docx(path):
    if not HAS_DOCX: return "[MISSING DEPENDENCY] Install 'python-docx'"
    try:
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    except: return ""

async def async_read_file(path, relative_path):
    print(f"   📄 Reading: {relative_path}...", end="\r") 
    ext = os.path.splitext(path)[1].lower()
    content = ""
    
    if ext == '.pdf': 
        try: content = "\n".join([p.extract_text() or "" for p in PdfReader(path).pages])
        except: content = ""
    elif ext == '.docx':
        content = await asyncio.to_thread(read_docx, path)
    elif ext == '.ipynb':
        try:
            with open(path, 'r', encoding='utf-8') as f: notebook = json.load(f)
            content = "\n".join(["".join(c['source']) for c in notebook.get('cells', []) if c['cell_type'] in ['code', 'markdown']])
        except: content = ""
    else: 
        content = await asyncio.to_thread(read_universal_text, path)
    
    return content

# --- Source Handler: GitHub Only ---

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
        if c and c.strip(): files_data[p] = {"content": c}

    print(f"\n✅ Total Loaded: {len(files_data)} items.")
    return files_data

# --- 2. Symbol Graph Summarizer (Advanced) ---

async def generate_symbol_graph(sem, filename, content, counter, total):
    async with sem:
        # We ask for "Exports" (what is defined) and "Dependencies" (what is used)
        prompt = f"""
        Analyze this code file for a Granular Dependency Graph.
        File: {filename}
        
        TASK:
        1. List 'exports': Major Classes and Functions defined in this file.
        2. List 'dependencies': External files AND the specific symbols (classes/funcs) used from them.
        
        Content Snippet (First 15k chars):
        {content[:15000]}
        
        Return JSON: {{ 
            "purpose": "Brief summary",
            "exports": ["class UserManager", "def validate_email"],
            "dependencies": [
                {{ "filename": "database.py", "symbols_used": ["class DBConnection", "def connect"] }}
            ]
        }}
        """
        try:
            res = await safe_chat_completion(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(res.choices[0].message.content)
            
            if "exports" not in data: data["exports"] = []
            if "dependencies" not in data: data["dependencies"] = []
            
            counter[0] += 1
            print(f"   [ {counter[0]}/{total} ] Mapped Symbols: {os.path.basename(filename)}")
            return data
        except Exception as e:
            counter[0] += 1
            return {"purpose": "Error", "exports": [], "dependencies": []}

async def summarizer_agent(files_data):
    print("\n🕵️  Summarizer Agent: Building Symbol Graph...")
    if os.path.exists(SUMMARIES_DIR): shutil.rmtree(SUMMARIES_DIR, onerror=handle_remove_readonly)
    os.makedirs(SUMMARIES_DIR)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    counter, total = [0], len(files_data)
    
    tasks = [generate_symbol_graph(sem, f, d['content'], counter, total) for f, d in files_data.items()]
    results = await asyncio.gather(*tasks)

    knowledge_base = {}
    for filename, result in zip(files_data.keys(), results):
        knowledge_base[filename] = result

    with open(KNOWLEDGE_BASE_FILE, 'w') as f: json.dump(knowledge_base, f, indent=2)
    return knowledge_base

# --- 3. Reframer Agent ---
async def reframer_agent(user_query, chat_history):
    print("🧠 Reframer: Clarifying intent using history...", flush=True)
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
    print(f"   ↳ Resolved Query: \"{new_query}\"")
    return new_query

# --- 4. Deep Graph Selector & Slicer ---

async def extract_relevant_code(filename, full_content, symbols_needed):
    """
    Uses LLM to slice the file and return ONLY the requested functions/classes.
    """
    if not symbols_needed:
        # If explicitly no symbols, maybe just return headers.
        return f"--- Snippet from {filename} (Headers Only) ---\n{full_content[:1000]}\n...(truncated)...\n"

    prompt = f"""
    You are a Code Slicer. 
    File: {filename}
    Target Symbols: {json.dumps(list(symbols_needed))}
    
    TASK:
    Return ONLY the code blocks (function definitions, class definitions) for the Target Symbols.
    Include necessary imports at the top.
    DO NOT summarize. Return actual code.
    
    Code Context:
    {full_content[:40000]} 
    """
    
    try:
        res = await safe_chat_completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return f"--- Snippet from {filename} (Filtered for: {', '.join(symbols_needed)}) ---\n{res.choices[0].message.content}\n"
    except:
        return f"--- {filename} ---\n{full_content[:5000]}\n" 

async def selector_agent(technical_query, knowledge_base, files_data):
    print(f"🗂️  Selector: Tracing Recursive Dependency Chain (Depth {MAX_RECURSION_DEPTH})...", flush=True)
    
    # 1. Identify Seeds
    index_view = {k: {"purpose": v.get('purpose'), "exports": v.get('exports')} for k, v in knowledge_base.items()}
    
    prompt = f"""
    Identify 2-3 Seed Files for: "{technical_query}"
    Look at 'exports' to find the code owners.
    Index: {json.dumps(index_view)}
    Return JSON: {{ "seed_files": ["main.py", "logic.py"] }}
    """
    
    res = await safe_chat_completion(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    seed_files = json.loads(res.choices[0].message.content).get('seed_files', [])
    print(f"   🌱 Seeds: {seed_files}")

    # "requirements" maps: Real Filename -> Set of Symbols Needed
    # If Set is "ALL", we load the full file (only for seeds).
    requirements: Dict[str, Union[str, Set[str]]] = {} 
    
    # Queue for BFS: List of Real Filenames
    queue = []

    # Initialize Seeds
    for seed in seed_files:
        real_seed = next((f for f in files_data.keys() if seed in f), None)
        if real_seed:
            requirements[real_seed] = "ALL"  # Seeds are loaded fully (change to set() if you want seeds sliced too)
            queue.append(real_seed)

    # 2. BFS Traversal (Up to Depth 5)
    current_depth = 0
    while queue and current_depth < MAX_RECURSION_DEPTH:
        # print(f"      📍 Depth {current_depth}: Processing {len(queue)} files...")
        next_queue = []
        
        for current_file in queue:
            if current_file not in knowledge_base: continue
            
            # Get dependencies of current file
            deps = knowledge_base[current_file].get("dependencies", [])
            
            for dep in deps:
                dep_name_ref = dep.get("filename")
                symbols_needed = set(dep.get("symbols_used", []))
                
                # Find valid file
                real_dep_file = next((f for f in files_data.keys() if dep_name_ref in f), None)
                if not real_dep_file: continue
                
                # If we haven't seen this file, add it
                if real_dep_file not in requirements:
                    requirements[real_dep_file] = symbols_needed
                    next_queue.append(real_dep_file)
                else:
                    # If we have seen it, MERGE symbols (unless it's already ALL)
                    if requirements[real_dep_file] != "ALL":
                        requirements[real_dep_file].update(symbols_needed)

        queue = next_queue
        current_depth += 1

    # 3. Generate Context (Slicing)
    final_context = []
    current_chars = 0
    
    # Sort files to put seeds first
    sorted_files = sorted(requirements.keys(), key=lambda k: 0 if requirements[k] == "ALL" else 1)

    print(f"      ✂️  Slicing {len(sorted_files)} files for context...")
    
    for fname in sorted_files:
        if current_chars > MAX_CONTEXT_CHARS:
            print(f"   ⚠️ Context limit reached. Stopping at {fname}")
            break
            
        reqs = requirements[fname]
        content = files_data[fname]['content']
        
        if reqs == "ALL":
            # Full Content (Seed)
            chunk = f"=== FOCUS FILE: {fname} ===\n{content}\n"
        else:
            # Sliced Content (Dependency)
            symbol_list = list(reqs)
            chunk = await extract_relevant_code(fname, content, symbol_list)
            
        final_context.append(chunk)
        current_chars += len(chunk)

    return final_context

# --- 5. Answering Agent ---
async def answering_agent(user_query, context_strings):
    print("📝 Answering Agent: Generating response...", flush=True)
    
    full_context = "\n".join(context_strings)
    
    res = await safe_chat_completion(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a senior developer. Answer based strictly on the provided Code Context. Cite which file logic comes from."},
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
        
        if not gh_input.strip():
            print("❌ No sources provided. Exiting.")
            return

        files_data = await ingest_sources(gh_input)
        
        if not files_data: 
            print("❌ No valid data loaded.")
            return

        # 1. Summarize (Symbol Level)
        knowledge_base = await summarizer_agent(files_data)
        
        while True:
            query = input("\n" + "-"*40 + "\n(Type 'exit' to quit) Question: ")
            if query.lower() in ['exit', 'quit']: break
            
            # 2. Reframe
            technical_query = await reframer_agent(query, chat_history)
            
            # 3. Select & Slice (Recursive Depth 5)
            context_strings = await selector_agent(technical_query, knowledge_base, files_data)
            
            if not context_strings:
                print("   ⚠️ No relevant code found.")
                continue

            # 4. Answer
            answer = await answering_agent(query, context_strings)
            
            print("\n" + "="*60 + f"\n✅ ANSWER:\n{answer}\n" + "="*60)
            
            # 5. Update History
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})

    except KeyboardInterrupt: print("\n⚠️ Interrupted.")
    except Exception as e: print(f"\n❌ Error: {e}")
    finally: perform_cleanup()

if __name__ == "__main__":
    asyncio.run(main())