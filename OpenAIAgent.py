import os
import json
import asyncio
import shutil
import stat
import errno
import sys
import re
import gdown
from typing import List, Dict, Optional, Set
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

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not found in .env. Please set it or enter it below.")
    # OPENAI_API_KEY = "sk-..." 

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Paths & Limits
TEMP_DIR = "./temp_session_data"
SUMMARIES_DIR = "./repo_summaries"
KNOWLEDGE_BASE_FILE = "session_knowledge.json"

MAX_CONCURRENCY = 50 
MAX_CONTEXT_CHARS = 200000 

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

# --- Source Handlers ---
async def handle_github_repo(url, source_id):
    repo_path = os.path.join(TEMP_DIR, f"repo_{source_id}")
    print(f"🔄 Cloning GitHub Repo: {url}...")
    try:
        await asyncio.to_thread(Repo.clone_from, url, repo_path)
        return repo_path
    except Exception as e:
        print(f"❌ Git Clone Failed: {e}")
        return None

async def handle_google_drive(url, source_id):
    output_path = os.path.join(TEMP_DIR, f"gdrive_{source_id}")
    os.makedirs(output_path, exist_ok=True)
    file_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
    file_id = file_id_match.group(1) if file_id_match else None
    
    if not file_id: return None

    final_url = url
    is_native_doc = False
    
    if "/document/d/" in url:
        final_url = f"https://docs.google.com/document/d/{file_id}/export?format=pdf"
        is_native_doc = True
    elif "/presentation/d/" in url:
        final_url = f"https://docs.google.com/presentation/d/{file_id}/export/pdf"
        is_native_doc = True
    elif "/spreadsheets/d/" in url:
        final_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=pdf"
        is_native_doc = True

    try:
        output_file = os.path.join(output_path, f"doc_{source_id}.pdf" if is_native_doc else f"file_{source_id}")
        await asyncio.to_thread(gdown.download, final_url, output_file, quiet=True, fuzzy=True)
        return output_path
    except Exception as e:
        print(f"❌ Drive Download Error: {e}")
        return None

async def ingest_sources(source_inputs: List[str]):
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR, onerror=handle_remove_readonly)
    os.makedirs(TEMP_DIR, exist_ok=True)
    tasks, paths = [], []
    
    for i, source in enumerate(source_inputs):
        source = source.strip()
        local_dir = None
        if "github.com" in source: local_dir = await handle_github_repo(source, i)
        elif "drive.google.com" in source or "docs.google.com" in source: local_dir = await handle_google_drive(source, i)
        else: continue
            
        if local_dir:
            for root, _, files in os.walk(local_dir):
                if ".git" in root: continue
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, TEMP_DIR)
                    if is_valid_file(file):
                        tasks.append(async_read_file(full_path, rel_path))
                        paths.append(rel_path)

    print(f"\n📖 Processing {len(tasks)} gathered files...")
    results = await asyncio.gather(*tasks)
    files_data = {}
    for p, c in zip(paths, results):
        if c and c.strip(): files_data[p] = {"content": c}
    print(f"\n✅ Total Loaded: {len(files_data)} valid files.")
    return files_data

# --- 2. Strict Graph Summarizer ---
async def generate_rich_summary(sem, filename, content, counter, total):
    async with sem:
        prompt = f"""
        Analyze this file for a Code Knowledge Graph.
        File: {filename}
        
        TASK:
        1. Summarize purpose.
        2. Extract STRICT Functional Dependencies (Imports/Inheritance).
        
        CRITICAL RULES:
        - ONLY list code files that are imported.
        - IGNORE doc files (.md, .txt) unless strictly relevant.
        
        Content Snippet:
        {content[:15000]}
        
        Return JSON: {{ 
            "dense_summary": {{ 
                "purpose": "string", 
                "dependencies": ["utils.py", "models/user.py"] 
            }} 
        }}
        """
        try:
            res = await safe_chat_completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(res.choices[0].message.content)
            if "dense_summary" not in data: data["dense_summary"] = {"purpose": "Error", "dependencies": []}
            counter[0] += 1
            print(f"   [ {counter[0]}/{total} ] Analyzed: {os.path.basename(filename)}")
            return data
        except:
            counter[0] += 1
            return {"dense_summary": {"purpose": "Error", "dependencies": []}}

async def summarizer_agent(files_data):
    print("\n🕵️  Summarizer Agent: Building Graph...")
    if os.path.exists(SUMMARIES_DIR): shutil.rmtree(SUMMARIES_DIR, onerror=handle_remove_readonly)
    os.makedirs(SUMMARIES_DIR)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    counter, total = [0], len(files_data)
    tasks = [generate_rich_summary(sem, f, d['content'], counter, total) for f, d in files_data.items()]
    results = await asyncio.gather(*tasks)

    knowledge_base = {}
    for filename, result in zip(files_data.keys(), results):
        knowledge_base[filename] = result['dense_summary']

    with open(KNOWLEDGE_BASE_FILE, 'w') as f: json.dump(knowledge_base, f, indent=2)
    return knowledge_base

# --- 3. Reframer Agent ---
async def reframer_agent(user_query, chat_history):
    print("🧠 Reframer: Clarifying intent using history...", flush=True)
    history_text = ""
    for turn in chat_history[-3:]: history_text += f"{turn['role'].upper()}: {turn['content']}\n"

    prompt = f"""
    You are a Technical Assistant. Rewrite the user's query into a precise search query.
    History: {history_text}
    Query: "{user_query}"
    Guidelines:
    1. Resolve pronouns (it, that).
    2. Detect topic shifts (new topic = ignore old history).
    3. Return ONLY the rewritten query.
    """
    res = await safe_chat_completion(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    new_query = res.choices[0].message.content.strip()
    print(f"   ↳ Resolved Query: \"{new_query}\"")
    return new_query

# --- 4. Intelligent Selector (With Pruning) ---

async def smart_filter_dependencies(query, parent_file, potential_deps, all_files):
    if not potential_deps: return []
    real_candidates = []
    for dep in potential_deps:
        matches = [f for f in all_files if dep in f or f.endswith(dep)]
        if matches: real_candidates.append(min(matches, key=len)) 
    
    if not real_candidates: return []

    prompt = f"""
    User Query: "{query}"
    File: "{parent_file}" imports {json.dumps(real_candidates)}
    
    Task: Return ONLY the imported files likely to contain logic relevant to the query.
    Return JSON: {{ "relevant_dependencies": ["file1.py"] }}
    """
    try:
        res = await safe_chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content).get('relevant_dependencies', [])
    except:
        return real_candidates 

async def selector_agent(technical_query, knowledge_base, files_data, feedback: Optional[str] = None):
    print(f"🗂️  Selector: Hunting for relevant files...", flush=True)
    
    index_view = {k: v['purpose'] for k, v in knowledge_base.items()}
    prompt = f"""
    Select 3-5 'Seed Files' for: "{technical_query}"
    Index: {json.dumps(index_view)}
    Return JSON: {{ "seed_files": ["file1", "file2"] }}
    """
    
    res = await safe_chat_completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    try:
        data = json.loads(res.choices[0].message.content)
        seed_files = [f for f in data.get('seed_files', []) if f in files_data]
        print(f"   🌱 Seeds: {seed_files}")
        
        final_selection_set = set(seed_files)
        all_filenames = list(files_data.keys())

        print("   🕸️  Traversing & Pruning Graph...")
        for seed in seed_files:
            if seed.endswith(('.md', '.txt', '.rst')): continue
            
            raw_deps = knowledge_base[seed].get("dependencies", [])
            relevant_deps = await smart_filter_dependencies(technical_query, seed, raw_deps, all_filenames)
            
            for dep in relevant_deps:
                if dep in files_data and dep not in final_selection_set:
                    print(f"      🔗 Linked (Relevant): {seed} -> {dep}")
                    final_selection_set.add(dep)
        
        final_selection = []
        current_chars = 0
        for fname in final_selection_set:
            f_len = len(files_data[fname]['content'])
            if current_chars + f_len > MAX_CONTEXT_CHARS:
                print(f"   ⚠️ Limit reached. Skipping {fname}")
                continue
            final_selection.append(fname)
            current_chars += f_len
            
        return final_selection
    except Exception as e:
        print(f"Selector Error: {e}")
        return []

# --- 5. Answering Agent ---
async def answering_agent(user_query, selected_files, files_data):
    print("📝 Answering Agent: Generating response...", flush=True)
    context = ""
    for fname in selected_files:
        if fname in files_data:
            context += f"\n=== {fname} ===\n{files_data[fname]['content']}\n"
            
    res = await safe_chat_completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior developer. Answer based strictly on the context."},
            {"role": "user", "content": f"Query: {user_query}\n\nContext:\n{context}"}
        ]
    )
    return res.choices[0].message.content, context

# --- Main Orchestrator ---
async def main():
    chat_history = []
    try:
        print("🔗 Enter sources (comma-separated):")
        raw_input = input("> ")
        if not raw_input: return
        sources = [s.strip() for s in raw_input.split(',')]
        files_data = await ingest_sources(sources)
        if not files_data: return

        # 1. Summarize
        knowledge_base = await summarizer_agent(files_data)
        
        while True:
            query = input("\n" + "-"*40 + "\n(Type 'exit' to quit) Question: ")
            if query.lower() in ['exit', 'quit']: break
            
            # 2. Reframe
            technical_query = await reframer_agent(query, chat_history)
            
            # 3. Select (Seed + Pruned Graph)
            selected_files = await selector_agent(technical_query, knowledge_base, files_data)
            
            # 4. Answer (No Evaluator Loop)
            answer, context = await answering_agent(query, selected_files, files_data)
            
            print("\n" + "="*60 + f"\n✅ ANSWER:\n{answer}\n" + "="*60)
            
            # 5. Update History
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})

    except KeyboardInterrupt: print("\n⚠️ Interrupted.")
    except Exception as e: print(f"\n❌ Error: {e}")
    finally: perform_cleanup()

if __name__ == "__main__":
    asyncio.run(main())