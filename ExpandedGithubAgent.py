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
import time
from typing import List, Dict, Optional, Set, Any, Tuple
from collections import defaultdict
from git import Repo
from openai import AsyncOpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
import tiktoken

# Tree-sitter imports with better error handling
HAS_TREE_SITTER = False
try:
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
    
    # Try to import language bindings
    TREE_SITTER_LANGS = {}
    
    try:
        import tree_sitter_python as tspython
        TREE_SITTER_LANGS['python'] = tspython
    except ImportError:
        pass
    
    try:
        import tree_sitter_javascript as tsjs
        TREE_SITTER_LANGS['javascript'] = tsjs
    except ImportError:
        pass
    
    try:
        import tree_sitter_typescript as tsts
        TREE_SITTER_LANGS['typescript'] = tsts
    except ImportError:
        pass
    
    try:
        import tree_sitter_java as tsjava
        TREE_SITTER_LANGS['java'] = tsjava
    except ImportError:
        pass
    
    try:
        import tree_sitter_go as tsgo
        TREE_SITTER_LANGS['go'] = tsgo
    except ImportError:
        pass
    
    try:
        import tree_sitter_rust as tsrust
        TREE_SITTER_LANGS['rust'] = tsrust
    except ImportError:
        pass
    
    try:
        import tree_sitter_cpp as tscpp
        TREE_SITTER_LANGS['cpp'] = tscpp
    except ImportError:
        pass
    
    try:
        import tree_sitter_c as tsc
        TREE_SITTER_LANGS['c'] = tsc
    except ImportError:
        pass
    
    try:
        import tree_sitter_bash as tsbash
        TREE_SITTER_LANGS['bash'] = tsbash
    except ImportError:
        pass
        
except ImportError:
    pass

# Vector DB imports
try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Jupyter notebook support
try:
    import nbformat
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Graph Configuration
MAX_RECURSION_DEPTH = 3
MAX_CONCURRENCY = 50 
MAX_CONTEXT_CHARS = 200000 
MAX_CONTEXT_TOKENS = 30000
EXTRACT_GLOBALS_LINES = 50  # Number of lines from top of file to extract
INCLUDE_FILE_HEADER = True  # Include file-level docstrings/comments

# Embedding Configuration
EMBEDDING_BATCH_SIZE = 100
TOP_K_SEEDS = 5
BACKWARD_TRAVERSAL_DEPTH = 3
INCLUDE_HIGH_LEVEL_CONTEXT = True
MAX_SUMMARY_TOKENS = 4000

# Cache Configuration
CACHE_DIR = "./pipeline_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "cache_metadata.json")
GRAPH_FILE = os.path.join(CACHE_DIR, "symbol_graph.json")

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not found in .env. Please set it.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
tokenizer = tiktoken.encoding_for_model(MODEL_NAME)

# Paths
TEMP_DIR = "./temp_session_data"

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

# --- Helper: Batch Embeddings ---
async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts."""
    if not texts:
        return []
    
    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return [[0.0] * 1536 for _ in texts]

# --- Helper: Token Counting ---
def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    return len(tokenizer.encode(text))

def truncate_to_token_budget(texts: List[str], budget: int) -> List[str]:
    """Truncate list of texts to fit within token budget."""
    result = []
    current_tokens = 0
    
    for text in texts:
        tokens = count_tokens(text)
        if current_tokens + tokens <= budget:
            result.append(text)
            current_tokens += tokens
        else:
            remaining = budget - current_tokens
            if remaining > 100:
                encoded = tokenizer.encode(text)[:remaining]
                result.append(tokenizer.decode(encoded) + "\n... [truncated]")
            break
    
    return result

# --- Helper: Route Normalization ---
def normalize_route(route: str) -> str:
    """Normalize API routes for matching (handle parameters)."""
    route = re.sub(r'/\d+', '/*', route)
    route = re.sub(r'/:\w+', '/*', route)
    route = re.sub(r'/\{[^}]+\}', '/*', route)
    route = re.sub(r'/\*+', '/*', route)
    return route.lower().strip()

def init_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"📁 Cache directory: {CACHE_DIR}")

# --- Cleanup ---
def perform_cleanup():
    print("\n🧹 Cleaning up temporary files...")
    if os.path.exists(TEMP_DIR):
        try: shutil.rmtree(TEMP_DIR, onerror=handle_remove_readonly)
        except: pass
    print("✅ Cleanup complete.")

# --- Repository Hashing for Cache Invalidation ---
def compute_repo_hash(repo_path: str) -> str:
    """Compute hash of all tracked files in repo for cache validation."""
    hasher = hashlib.md5()
    try:
        repo = Repo(repo_path)
        for item in repo.tree().traverse():
            if item.type == 'blob':
                hasher.update(item.path.encode())
                hasher.update(str(item.hexsha).encode())
    except:
        for root, dirs, files in os.walk(repo_path):
            for f in sorted(files):
                hasher.update(f.encode())
    return hasher.hexdigest()

# --- 1. Universal File Loader ---

def is_valid_file(filename):
    ALWAYS_KEEP_NAMES = {
        'dockerfile', 'makefile', 'gemfile', 'jenkinsfile', 'procfile', 
        'requirements.txt', 'package.json', 'cargo.toml', 'go.mod', 'pom.xml',
        'tsconfig.json', 'go.sum', 'package-lock.json'
    }
    ALLOWED_EXTENSIONS = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.h', '.cs', 
        '.go', '.rs', '.rb', '.php', '.c', '.cc', '.hpp', '.sh', '.bash',
        '.ipynb', '.ino'  # Added Jupyter notebooks and Arduino files
    }
    name = os.path.basename(filename).lower()
    ext = os.path.splitext(filename)[1].lower()
    return name in ALWAYS_KEEP_NAMES or ext in ALLOWED_EXTENSIONS

def read_universal_text(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f: 
            return f.read()
    except: 
        return ""

def extract_notebook_code(notebook_path: str) -> str:
    """Extract code cells from Jupyter notebook."""
    if not HAS_NBFORMAT:
        return read_universal_text(notebook_path)
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        code_cells = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                code_cells.append(f"# Cell {i+1}\n{cell.source}")
        
        return "\n\n".join(code_cells)
    except Exception as e:
        print(f"⚠️ Error reading notebook {notebook_path}: {e}")
        return read_universal_text(notebook_path)

async def async_read_file(path, relative_path):
    """Read file with special handling for notebooks."""
    if path.endswith('.ipynb'):
        return await asyncio.to_thread(extract_notebook_code, path)
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
    if os.path.exists(TEMP_DIR): 
        shutil.rmtree(TEMP_DIR, onerror=handle_remove_readonly)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    git_urls = [s.strip() for s in github_inputs.split(',') if s.strip()]
    
    dir_tasks = []
    for i, url in enumerate(git_urls):
        dir_tasks.append(handle_github_repo(url, i))
        
    repo_results = await asyncio.gather(*dir_tasks)
    
    multi_repo_data = {}
    read_tasks = []
    repo_hashes = {}
    
    for repo_path, repo_name in repo_results:
        if not repo_path: continue
        
        if repo_name not in multi_repo_data:
            multi_repo_data[repo_name] = {}
            repo_hashes[repo_name] = compute_repo_hash(repo_path)
            
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

    print(f"✅ Total Loaded: {total_files} files across {list(multi_repo_data.keys())}.")
    return multi_repo_data, repo_hashes

class GlobalExtractor:
    """
    Extracts global variables, constants, configurations, and defines from files.
    CRITICAL for understanding code context beyond just function definitions.
    """

    def extract_globals_from_content(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Extract global-level declarations.
        Returns: {header, imports, constants, globals, config, defines}
        """
        ext = os.path.splitext(filename)[1].lower()
        
        result = {
            "header": "",
            "imports": [],
            "constants": [],
            "globals": [],
            "config": [],
            "defines": []
        }
        
        lines = content.split('\n')
        
        if INCLUDE_FILE_HEADER:
            result["header"] = self._extract_header(lines, ext)
        
        if ext == '.py':
            result.update(self._extract_python_globals(content, lines))
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            result.update(self._extract_js_globals(content, lines))
        elif ext in ['.cpp', '.cc', '.h', '.hpp', '.c', '.ino']:
            result.update(self._extract_cpp_globals(content, lines))
        elif ext in ['.java']:
            result.update(self._extract_java_globals(content, lines))
        elif ext in ['.go']:
            result.update(self._extract_go_globals(content, lines))
        else:
            result.update(self._extract_generic_globals(content, lines))
        
        return result

    def _extract_header(self, lines: List[str], ext: str) -> str:
        """Extract file-level documentation."""
        header_lines = []
        
        if ext == '.py':
            if len(lines) > 0 and lines[0].strip().startswith('"""'):
                for line in lines[:20]:
                    header_lines.append(line)
                    if line.strip().endswith('"""') and len(header_lines) > 1:
                        break
        
        elif ext in ['.cpp', '.c', '.h', '.hpp', '.ino', '.js', '.jsx', '.ts', '.tsx']:
            for line in lines[:30]:
                stripped = line.strip()
                if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                    header_lines.append(line)
                elif header_lines and not stripped:
                    continue
                else:
                    break
        
        return '\n'.join(header_lines)

    def _extract_python_globals(self, content: str, lines: List[str]) -> Dict:
        """Extract Python globals."""
        imports, constants, globals_vars, config = [], [], [], []
        
        try:
            tree = ast.parse(content)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.get_source_segment(content, node))
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            var_line = ast.get_source_segment(content, node)
                            if var_name.isupper():
                                constants.append(var_line)
                            elif 'CONFIG' in var_name or 'SETTINGS' in var_name:
                                config.append(var_line)
                            else:
                                globals_vars.append(var_line)
        except:
            for line in lines[:EXTRACT_GLOBALS_LINES]:
                if re.match(r'^import\s+|^from\s+', line.strip()):
                    imports.append(line)
                elif re.match(r'^[A-Z_]+\s*=', line.strip()):
                    constants.append(line)
                elif re.match(r'^[a-z_]\w*\s*=', line.strip()):
                    globals_vars.append(line)
        
        return {"imports": imports, "constants": constants, "globals": globals_vars, "config": config}

    def _extract_js_globals(self, content: str, lines: List[str]) -> Dict:
        """Extract JavaScript/TypeScript globals."""
        imports, constants, globals_vars, config = [], [], [], []
        
        for line in lines[:EXTRACT_GLOBALS_LINES]:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('require('):
                imports.append(line)
            elif re.match(r'const\s+[A-Z_]+', stripped):
                constants.append(line)
            elif 'config' in stripped.lower() or 'settings' in stripped.lower():
                config.append(line)
            elif re.match(r'(let|var|const)\s+\w+', stripped):
                globals_vars.append(line)
        
        return {"imports": imports, "constants": constants, "globals": globals_vars, "config": config}

    def _extract_cpp_globals(self, content: str, lines: List[str]) -> Dict:
        """Extract C/C++ globals and defines."""
        imports, constants, globals_vars, defines = [], [], [], []
        
        for line in lines[:EXTRACT_GLOBALS_LINES]:
            stripped = line.strip()
            if stripped.startswith('#include'):
                imports.append(line)
            elif stripped.startswith('#define'):
                defines.append(line)
            elif 'const ' in stripped:
                constants.append(line)
            elif re.match(r'(extern|static)?\s*(int|float|double|char|bool|void|uint\w*|String)', stripped):
                if not stripped.endswith(';'):
                    idx = lines.index(line)
                    multi = line
                    for i in range(1, 5):
                        if idx + i < len(lines):
                            multi += '\n' + lines[idx + i]
                            if ';' in lines[idx + i]:
                                break
                    globals_vars.append(multi)
                else:
                    globals_vars.append(line)
        
        return {"imports": imports, "constants": constants, "globals": globals_vars, "defines": defines}

    def _extract_java_globals(self, content: str, lines: List[str]) -> Dict:
        """Extract Java globals."""
        imports, constants, globals_vars = [], [], []
        in_class = False
        
        for line in lines[:EXTRACT_GLOBALS_LINES]:
            stripped = line.strip()
            if stripped.startswith('import '):
                imports.append(line)
            elif 'class ' in stripped:
                in_class = True
            elif in_class and 'static final' in stripped:
                constants.append(line)
            elif in_class and 'static ' in stripped:
                globals_vars.append(line)
        
        return {"imports": imports, "constants": constants, "globals": globals_vars}

    def _extract_go_globals(self, content: str, lines: List[str]) -> Dict:
        """Extract Go globals."""
        imports, constants, globals_vars = [], [], []
        
        for line in lines[:EXTRACT_GLOBALS_LINES]:
            stripped = line.strip()
            if stripped.startswith('import '):
                imports.append(line)
            elif stripped.startswith('const '):
                constants.append(line)
            elif stripped.startswith('var '):
                globals_vars.append(line)
        
        return {"imports": imports, "constants": constants, "globals": globals_vars}

    def _extract_generic_globals(self, content: str, lines: List[str]) -> Dict:
        """Generic extraction."""
        globals_vars = []
        for line in lines[:EXTRACT_GLOBALS_LINES]:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('//'):
                globals_vars.append(line)
        return {"globals": globals_vars}

    def format_globals_for_context(self, globals_dict: Dict[str, List[str]]) -> str:
        """Format extracted globals into readable context."""
        sections = []
        
        if globals_dict.get("header"):
            sections.append(f"# FILE HEADER\n{globals_dict['header']}\n")
        if globals_dict.get("imports"):
            sections.append(f"# IMPORTS\n" + "\n".join(globals_dict['imports']) + "\n")
        if globals_dict.get("defines"):
            sections.append(f"# DEFINES\n" + "\n".join(globals_dict['defines']) + "\n")
        if globals_dict.get("constants"):
            sections.append(f"# CONSTANTS\n" + "\n".join(globals_dict['constants']) + "\n")
        if globals_dict.get("config"):
            sections.append(f"# CONFIGURATION\n" + "\n".join(globals_dict['config']) + "\n")
        if globals_dict.get("globals"):
            sections.append(f"# GLOBAL VARIABLES\n" + "\n".join(globals_dict['globals']) + "\n")
        
        return "\n".join(sections) if sections else ""

# --- 2. Enhanced Tree-Sitter Parser ---

class TreeSitterParser:
    """
    Production-grade parser using Tree-sitter with fallback to regex.
    """
    
    def __init__(self):
        self.parsers = {}
        self.languages = {}
        self.global_extractor = GlobalExtractor()
        
        if not HAS_TREE_SITTER:
            print("⚠️ Tree-sitter not available, using regex fallback")
            return
        
        # Initialize available languages
        self._init_languages()
        
        # Regex fallback patterns for unsupported languages
        self.REGEX_PATTERNS = {
            'cpp': [
                r'(?:void|int|float|double|bool|char|auto)\s+(\w+)\s*\([^)]*\)\s*\{',
                r'(\w+)\s*::\s*(\w+)\s*\([^)]*\)\s*\{',  # Class methods
                r'class\s+(\w+)',
            ],
            'c': [
                r'(?:void|int|float|double|char|static|extern)\s+(\w+)\s*\([^)]*\)\s*\{',
            ],
            'sh': [
                r'function\s+(\w+)\s*\(\s*\)\s*\{',
                r'(\w+)\s*\(\s*\)\s*\{',
            ],
            'bash': [
                r'function\s+(\w+)\s*\(\s*\)\s*\{',
                r'(\w+)\s*\(\s*\)\s*\{',
            ],
            'ino': [  # Arduino (C++ based)
                r'(?:void|int|float|double|bool|char)\s+(\w+)\s*\([^)]*\)\s*\{',
            ],
        }
        
        # API patterns
        self.API_CALL_PATTERNS = [
            r'fetch\s*\(\s*["\']([^"\']+)["\']',
            r'axios\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
            r'http\.(?:Get|Post|Put|Delete)\s*\(\s*["\']([^"\']+)["\']',
            r'requests\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
        ]
        
        self.API_ROUTE_PATTERNS = [
            r'@app\.(?:get|post|put|delete|route)\s*\(\s*["\']([^"\']+)["\']',
            r'@router\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
            r'@(?:Get|Post|Put|Delete)Mapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']',
            r'app\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
            r'router\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
        ]

    def _init_languages(self):
        """Initialize Tree-sitter language parsers."""
        try:
            # Python
            if 'python' in TREE_SITTER_LANGS:
                PY_LANGUAGE = Language(TREE_SITTER_LANGS['python'].language())
                self.languages['python'] = PY_LANGUAGE
                self.parsers['.py'] = Parser(PY_LANGUAGE)
            
            # JavaScript
            if 'javascript' in TREE_SITTER_LANGS:
                JS_LANGUAGE = Language(TREE_SITTER_LANGS['javascript'].language())
                self.languages['javascript'] = JS_LANGUAGE
                js_parser = Parser(JS_LANGUAGE)
                self.parsers['.js'] = js_parser
                self.parsers['.jsx'] = js_parser
            
            # TypeScript
            if 'typescript' in TREE_SITTER_LANGS:
                TS_LANGUAGE = Language(TREE_SITTER_LANGS['typescript'].language_typescript())
                self.languages['typescript'] = TS_LANGUAGE
                self.parsers['.ts'] = Parser(TS_LANGUAGE)
                
                TSX_LANGUAGE = Language(TREE_SITTER_LANGS['typescript'].language_tsx())
                self.parsers['.tsx'] = Parser(TSX_LANGUAGE)
            
            # Java
            if 'java' in TREE_SITTER_LANGS:
                JAVA_LANGUAGE = Language(TREE_SITTER_LANGS['java'].language())
                self.languages['java'] = JAVA_LANGUAGE
                self.parsers['.java'] = Parser(JAVA_LANGUAGE)
            
            # Go
            if 'go' in TREE_SITTER_LANGS:
                GO_LANGUAGE = Language(TREE_SITTER_LANGS['go'].language())
                self.languages['go'] = GO_LANGUAGE
                self.parsers['.go'] = Parser(GO_LANGUAGE)
            
            # Rust
            if 'rust' in TREE_SITTER_LANGS:
                RUST_LANGUAGE = Language(TREE_SITTER_LANGS['rust'].language())
                self.languages['rust'] = RUST_LANGUAGE
                self.parsers['.rs'] = Parser(RUST_LANGUAGE)
            
            # C++
            if 'cpp' in TREE_SITTER_LANGS:
                CPP_LANGUAGE = Language(TREE_SITTER_LANGS['cpp'].language())
                self.languages['cpp'] = CPP_LANGUAGE
                cpp_parser = Parser(CPP_LANGUAGE)
                self.parsers['.cpp'] = cpp_parser
                self.parsers['.cc'] = cpp_parser
                self.parsers['.hpp'] = cpp_parser
                self.parsers['.ino'] = cpp_parser  # Arduino
            
            # C
            if 'c' in TREE_SITTER_LANGS:
                C_LANGUAGE = Language(TREE_SITTER_LANGS['c'].language())
                self.languages['c'] = C_LANGUAGE
                c_parser = Parser(C_LANGUAGE)
                self.parsers['.c'] = c_parser
                self.parsers['.h'] = c_parser
            
            # Bash
            if 'bash' in TREE_SITTER_LANGS:
                BASH_LANGUAGE = Language(TREE_SITTER_LANGS['bash'].language())
                self.languages['bash'] = BASH_LANGUAGE
                bash_parser = Parser(BASH_LANGUAGE)
                self.parsers['.sh'] = bash_parser
                self.parsers['.bash'] = bash_parser
            
            if self.parsers:
                print(f"✅ Tree-sitter initialized for: {list(set(ext for ext in self.parsers.keys()))}")
            
        except Exception as e:
            print(f"⚠️ Tree-sitter initialization error: {e}")
            print("   Falling back to regex parsing for all languages")

    def parse(self, filename: str, content: str) -> Dict[str, Any]:
        """Parse file and extract symbols WITH enhanced global extraction."""
        ext = os.path.splitext(filename)[1].lower()
        # FIRST: Extract globals using enhanced extractor
        globals_data = self.global_extractor.extract_globals_from_content(content, filename)
        formatted_globals = self.global_extractor.format_globals_for_context(globals_data)

        # THEN: Extract functions using tree-sitter or regex
        parser = self.parsers.get(ext)
        if parser:
            try:
                tree = parser.parse(bytes(content, "utf8"))
                result = self._extract_symbols(tree.root_node, content, ext)
            except Exception as e:
                print(f"⚠️ Tree-sitter parse error in {filename}: {e}, using regex fallback")
                result = self._parse_regex_fallback(content, ext, filename)
        else:
            result = self._parse_regex_fallback(content, ext, filename)

        # COMBINE: Replace simple imports with rich global context
        result['globals'] = formatted_globals  # This now includes MUCH more!
        result['globals_data'] = globals_data  # Store structured data too

        return result

    def _extract_symbols(self, root_node, content: str, ext: str) -> Dict[str, Any]:
        """Extract functions, classes, and imports from AST."""
        if ext == '.py':
            return self._extract_python(root_node, content)
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            return self._extract_javascript(root_node, content)
        elif ext == '.java':
            return self._extract_java(root_node, content)
        elif ext == '.go':
            return self._extract_go(root_node, content)
        elif ext == '.rs':
            return self._extract_rust(root_node, content)
        elif ext in ['.cpp', '.cc', '.hpp', '.c', '.h', '.ino']:
            return self._extract_cpp(root_node, content)
        elif ext in ['.sh', '.bash']:
            return self._extract_bash(root_node, content)
        else:
            return {"nodes": [], "imports": [], "globals": ""}

    def _get_text(self, node, content: str) -> str:
        """Extract text from a node."""
        return content[node.start_byte:node.end_byte]

    def _extract_python(self, root, content: str) -> Dict[str, Any]:
        """Extract Python symbols."""
        nodes = []
        imports = []
        
        def visit(node, namespace=""):
            if node.type == 'import_statement' or node.type == 'import_from_statement':
                imports.append(self._get_text(node, content))
            
            elif node.type == 'function_definition':
                func_name_node = node.child_by_field_name('name')
                if func_name_node:
                    name = self._get_text(func_name_node, content)
                    full_name = f"{namespace}.{name}" if namespace else name
                    code = self._get_text(node, content)
                    calls = self._extract_calls_python(node, content)
                    api_route = self._extract_python_route(node, content)
                    api_calls = self._extract_api_calls(code)
                    
                    nodes.append({
                        "name": full_name,
                        "type": "function",
                        "code": code,
                        "calls": calls,
                        "api_route": api_route,
                        "api_outbound": api_calls
                    })
            
            elif node.type == 'class_definition':
                class_name_node = node.child_by_field_name('name')
                if class_name_node:
                    class_name = self._get_text(class_name_node, content)
                    new_namespace = f"{namespace}.{class_name}" if namespace else class_name
                    for child in node.children:
                        visit(child, new_namespace)
                    return
            
            for child in node.children:
                visit(child, namespace)
        
        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_calls_python(self, func_node, content: str) -> List[str]:
        """Extract function calls from Python function."""
        calls = set()
        
        def visit(node):
            if node.type == 'call':
                func_node = node.child_by_field_name('function')
                if func_node:
                    if func_node.type == 'identifier':
                        calls.add(self._get_text(func_node, content))
                    elif func_node.type == 'attribute':
                        attr = func_node.child_by_field_name('attribute')
                        if attr:
                            calls.add(self._get_text(attr, content))
            
            for child in node.children:
                visit(child)
        
        visit(func_node)
        return list(calls)

    def _extract_python_route(self, func_node, content: str) -> Optional[str]:
        """Extract API route from Python decorators."""
        prev = func_node.prev_sibling
        while prev and prev.type == 'decorator':
            decorator_text = self._get_text(prev, content)
            for pattern in self.API_ROUTE_PATTERNS:
                match = re.search(pattern, decorator_text)
                if match:
                    return normalize_route(match.group(1))
            prev = prev.prev_sibling
        return None

    def _extract_javascript(self, root, content: str) -> Dict[str, Any]:
        """Extract JavaScript/TypeScript symbols."""
        nodes = []
        imports = []
        
        def visit(node, namespace=""):
            if node.type in ['import_statement', 'import_clause']:
                imports.append(self._get_text(node, content))
            
            elif node.type == 'function_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_text(name_node, content)
                    code = self._get_text(node, content)
                    calls = self._extract_calls_js(node, content)
                    api_calls = self._extract_api_calls(code)
                    
                    nodes.append({
                        "name": name,
                        "type": "function",
                        "code": code,
                        "calls": calls,
                        "api_route": None,
                        "api_outbound": api_calls
                    })
            
            elif node.type == 'lexical_declaration':
                for child in node.children:
                    if child.type == 'variable_declarator':
                        name_node = child.child_by_field_name('name')
                        value_node = child.child_by_field_name('value')
                        if name_node and value_node:
                            if value_node.type in ['arrow_function', 'function']:
                                name = self._get_text(name_node, content)
                                code = self._get_text(child, content)
                                calls = self._extract_calls_js(value_node, content)
                                api_calls = self._extract_api_calls(code)
                                
                                nodes.append({
                                    "name": name,
                                    "type": "function",
                                    "code": code,
                                    "calls": calls,
                                    "api_route": None,
                                    "api_outbound": api_calls
                                })
            
            elif node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = self._get_text(name_node, content)
                    body = node.child_by_field_name('body')
                    if body:
                        for child in body.children:
                            if child.type == 'method_definition':
                                method_name_node = child.child_by_field_name('name')
                                if method_name_node:
                                    method_name = self._get_text(method_name_node, content)
                                    full_name = f"{class_name}.{method_name}"
                                    code = self._get_text(child, content)
                                    calls = self._extract_calls_js(child, content)
                                    api_calls = self._extract_api_calls(code)
                                    
                                    nodes.append({
                                        "name": full_name,
                                        "type": "method",
                                        "code": code,
                                        "calls": calls,
                                        "api_route": None,
                                        "api_outbound": api_calls
                                    })
            
            for child in node.children:
                visit(child, namespace)
        
        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_calls_js(self, node, content: str) -> List[str]:
        """Extract function calls from JS/TS code."""
        calls = set()
        
        def visit(n):
            if n.type == 'call_expression':
                func = n.child_by_field_name('function')
                if func:
                    if func.type == 'identifier':
                        calls.add(self._get_text(func, content))
                    elif func.type == 'member_expression':
                        prop = func.child_by_field_name('property')
                        if prop:
                            calls.add(self._get_text(prop, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_java(self, root, content: str) -> Dict[str, Any]:
        """Extract Java symbols."""
        nodes = []
        imports = []
        
        def visit(node, class_context=""):
            if node.type == 'import_declaration':
                imports.append(self._get_text(node, content))
            
            elif node.type == 'method_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_text(name_node, content)
                    full_name = f"{class_context}.{name}" if class_context else name
                    code = self._get_text(node, content)
                    calls = self._extract_calls_java(node, content)
                    api_route = self._extract_spring_route(node, content)
                    api_calls = self._extract_api_calls(code)
                    
                    nodes.append({
                        "name": full_name,
                        "type": "method",
                        "code": code,
                        "calls": calls,
                        "api_route": api_route,
                        "api_outbound": api_calls
                    })
            
            elif node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = self._get_text(name_node, content)
                    body = node.child_by_field_name('body')
                    if body:
                        for child in body.children:
                            visit(child, class_name)
                    return
            
            for child in node.children:
                visit(child, class_context)
        
        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_calls_java(self, node, content: str) -> List[str]:
        """Extract method calls from Java code."""
        calls = set()
        
        def visit(n):
            if n.type == 'method_invocation':
                name_node = n.child_by_field_name('name')
                if name_node:
                    calls.add(self._get_text(name_node, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_spring_route(self, method_node, content: str) -> Optional[str]:
        """Extract Spring @RequestMapping, @GetMapping, etc."""
        prev = method_node.prev_sibling
        while prev:
            if prev.type == 'marker_annotation' or prev.type == 'annotation':
                annot_text = self._get_text(prev, content)
                patterns = [
                    r'@(?:Get|Post|Put|Delete)Mapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']',
                    r'@RequestMapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']'
                ]
                for pattern in patterns:
                    match = re.search(pattern, annot_text)
                    if match:
                        return normalize_route(match.group(1))
            prev = prev.prev_sibling
        return None

    def _extract_go(self, root, content: str) -> Dict[str, Any]:
        """Extract Go symbols."""
        nodes = []
        imports = []
        
        def visit(node):
            if node.type == 'import_declaration':
                imports.append(self._get_text(node, content))
            
            elif node.type == 'function_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_text(name_node, content)
                    code = self._get_text(node, content)
                    calls = self._extract_calls_go(node, content)
                    api_calls = self._extract_api_calls(code)
                    
                    nodes.append({
                        "name": name,
                        "type": "function",
                        "code": code,
                        "calls": calls,
                        "api_route": None,
                        "api_outbound": api_calls
                    })
            
            for child in node.children:
                visit(child)
        
        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_calls_go(self, node, content: str) -> List[str]:
        """Extract function calls from Go code."""
        calls = set()
        
        def visit(n):
            if n.type == 'call_expression':
                func = n.child_by_field_name('function')
                if func:
                    if func.type == 'identifier':
                        calls.add(self._get_text(func, content))
                    elif func.type == 'selector_expression':
                        field = func.child_by_field_name('field')
                        if field:
                            calls.add(self._get_text(field, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_rust(self, root, content: str) -> Dict[str, Any]:
        """Extract Rust symbols."""
        nodes = []
        imports = []
        
        def visit(node):
            if node.type == 'use_declaration':
                imports.append(self._get_text(node, content))
            
            elif node.type == 'function_item':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_text(name_node, content)
                    code = self._get_text(node, content)
                    calls = self._extract_calls_rust(node, content)
                    api_calls = self._extract_api_calls(code)
                    
                    nodes.append({
                        "name": name,
                        "type": "function",
                        "code": code,
                        "calls": calls,
                        "api_route": None,
                        "api_outbound": api_calls
                    })
            
            for child in node.children:
                visit(child)
        
        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_calls_rust(self, node, content: str) -> List[str]:
        """Extract function calls from Rust code."""
        calls = set()
        
        def visit(n):
            if n.type == 'call_expression':
                func = n.child_by_field_name('function')
                if func:
                    if func.type == 'identifier':
                        calls.add(self._get_text(func, content))
                    elif func.type == 'field_expression':
                        field = func.child_by_field_name('field')
                        if field:
                            calls.add(self._get_text(field, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_cpp(self, root, content: str) -> Dict[str, Any]:
        """Extract C/C++ symbols."""
        nodes = []
        imports = []
        
        def visit(node, namespace=""):
            if node.type == 'preproc_include':
                imports.append(self._get_text(node, content))
            
            elif node.type == 'function_definition':
                # Try to get function name from declarator
                declarator = node.child_by_field_name('declarator')
                if declarator:
                    name = self._extract_function_name_cpp(declarator, content)
                    if name:
                        full_name = f"{namespace}::{name}" if namespace else name
                        code = self._get_text(node, content)
                        calls = self._extract_calls_cpp(node, content)
                        api_calls = self._extract_api_calls(code)
                        
                        nodes.append({
                            "name": full_name,
                            "type": "function",
                            "code": code,
                            "calls": calls,
                            "api_route": None,
                            "api_outbound": api_calls
                        })
            
            elif node.type == 'class_specifier':
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = self._get_text(name_node, content)
                    body = node.child_by_field_name('body')
                    if body:
                        for child in body.children:
                            visit(child, class_name)
                    return
            
            for child in node.children:
                visit(child, namespace)
        
        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_function_name_cpp(self, declarator, content: str) -> Optional[str]:
        """Extract function name from C++ declarator."""
        if declarator.type == 'function_declarator':
            decl = declarator.child_by_field_name('declarator')
            if decl:
                if decl.type == 'identifier':
                    return self._get_text(decl, content)
                elif decl.type == 'field_identifier':
                    return self._get_text(decl, content)
                elif decl.type == 'qualified_identifier':
                    name = decl.child_by_field_name('name')
                    if name:
                        return self._get_text(name, content)
        elif declarator.type == 'identifier':
            return self._get_text(declarator, content)
        return None

    def _extract_calls_cpp(self, node, content: str) -> List[str]:
        """Extract function calls from C/C++ code."""
        calls = set()
        
        def visit(n):
            if n.type == 'call_expression':
                func = n.child_by_field_name('function')
                if func:
                    if func.type == 'identifier':
                        calls.add(self._get_text(func, content))
                    elif func.type == 'field_expression':
                        field = func.child_by_field_name('field')
                        if field:
                            calls.add(self._get_text(field, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_bash(self, root, content: str) -> Dict[str, Any]:
        """Extract Bash/Shell script symbols."""
        nodes = []
        imports = []
        
        def visit(node):
            if node.type == 'function_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_text(name_node, content)
                    code = self._get_text(node, content)
                    calls = self._extract_calls_bash(node, content)
                    
                    nodes.append({
                        "name": name,
                        "type": "function",
                        "code": code,
                        "calls": calls,
                        "api_route": None,
                        "api_outbound": []
                    })
            
            for child in node.children:
                visit(child)
        
        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": ""}

    def _extract_calls_bash(self, node, content: str) -> List[str]:
        """Extract function calls from Bash script."""
        calls = set()
        
        def visit(n):
            if n.type == 'command':
                name = n.child_by_field_name('name')
                if name and name.type == 'command_name':
                    first_child = name.children[0] if name.children else None
                    if first_child:
                        calls.add(self._get_text(first_child, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_api_calls(self, code_snippet: str) -> List[str]:
        """Extract API endpoint calls from code."""
        endpoints = []
        for pattern in self.API_CALL_PATTERNS:
            matches = re.findall(pattern, code_snippet)
            endpoints.extend([normalize_route(m) for m in matches])
        return list(set(endpoints))

    def _parse_regex_fallback(self, content: str, ext: str, filename: str) -> Dict[str, Any]:
        """Enhanced regex fallback parser."""
        nodes = []
        
        # Determine language key
        lang_key = ext.replace('.', '')
        if lang_key == 'ino':
            lang_key = 'cpp'  # Arduino uses C++
        
        patterns = self.REGEX_PATTERNS.get(lang_key, [
            r'function\s+(\w+)\s*\(',
            r'def\s+(\w+)\s*\(',
            r'fn\s+(\w+)\s*\(',
        ])
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    # Get the last captured group (the function name)
                    groups = match.groups()
                    name = groups[-1] if groups else match.group(1)
                    
                    code_snippet = "\n".join(lines[i:min(i+20, len(lines))])
                    
                    # Extract calls using basic patterns
                    call_pattern = r'(\w+)\s*\('
                    calls = list(set(re.findall(call_pattern, code_snippet)))
                    # Filter out common keywords
                    excludes = {'if', 'for', 'while', 'switch', 'catch', 'function', 'return', 'void', 'int', 'float'}
                    calls = [c for c in calls if c not in excludes]
                    
                    api_calls = self._extract_api_calls(code_snippet)
                    
                    nodes.append({
                        "name": name,
                        "type": "function",
                        "code": code_snippet,
                        "calls": calls,
                        "api_route": None,
                        "api_outbound": api_calls
                    })
                    break
        
        return {"nodes": nodes, "imports": [], "globals": ""}

# --- 3. Project Summarization System ---

class ProjectSummarizer:
    """
    Generates high-level summaries of files and the overall project architecture.
    Uses LLM to synthesize information for broader context.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.file_summaries = {}
        self.project_summary = ""

    async def summarize_file(self, filename: str, content: str, nodes: List[Dict]) -> str:
        """Generate a concise summary of a single file."""
        node_names = [n['name'] for n in nodes[:10]]
        node_str = ", ".join(node_names)
        
        prompt = f"""
Summarize the following file in 1-2 concise sentences. 
Focus on its primary responsibility in the system.
Filename: {filename}
Key Symbols: {node_str}
Content Preview: {content[:1000]}
"""
        try:
            res = await safe_chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = res.choices[0].message.content.strip()
            self.file_summaries[filename] = summary
            return summary
        except Exception as e:
            print(f"   ⚠️ Summary failed for {filename}: {e}")
            return f"Module containing {node_str}"

    async def generate_project_overview(self, multi_graph: Dict[str, Dict]) -> str:
        """Synthesize project-level architecture summary."""
        print("🧠 Generating project architectural overview...", flush=True)
        
        file_overviews = []
        for repo, graph in multi_graph.items():
            repo_files = set()
            for node_id, data in graph.items():
                repo_files.add(data['file'])
            
            # Use only a subset if too many
            subset = list(repo_files)[:30]
            summaries = [f"- {f}: {self.file_summaries.get(f, 'Source file')}" for f in subset]
            file_overviews.append(f"Repo [{repo}]:\n" + "\n".join(summaries))
        
        all_files_str = "\n\n".join(file_overviews)
        
        prompt = f"""
Given the following list of files and their short descriptions, provide a high-level architectural overview of the project.
Describe the main components, how they interact, and the overall purpose of the system.
Keep it under 3-4 paragraphs.

{all_files_str}
"""
        try:
            res = await safe_chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            self.project_summary = res.choices[0].message.content.strip()
            return self.project_summary
        except Exception as e:
            print(f"   ⚠️ Project overview failed: {e}")
            return "Multi-repository software project."

    def save(self, filepath: str):
        """Save summaries to disk."""
        data = {
            "file_summaries": self.file_summaries,
            "project_summary": self.project_summary
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load summaries from disk."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.file_summaries = data.get("file_summaries", {})
                self.project_summary = data.get("project_summary", "")

# --- 4. Import Resolution System ---

class ImportResolver:
    """Resolves imports to enable namespace-aware linking."""
    
    def __init__(self, repo_files: Dict[str, Dict]):
        self.repo_files = repo_files
        self.import_map = {}
        self.export_map = {}
        
    def build_maps(self, parsed_data: Dict[str, Dict]):
        """Build import and export maps for the repository."""
        
        # First pass: collect exports
        for filepath, data in parsed_data.items():
            exports = set()
            for node in data.get('nodes', []):
                exports.add(node['name'].split('.')[-1])
            self.export_map[filepath] = exports
        
        # Second pass: resolve imports
        for filepath, data in parsed_data.items():
            self.import_map[filepath] = {}
            
            for import_stmt in data.get('imports', []):
                resolved = self._resolve_import(filepath, import_stmt)
                if resolved:
                    for symbol, source in resolved.items():
                        self.import_map[filepath][symbol] = source
    
    def _resolve_import(self, current_file: str, import_stmt: str) -> Dict[str, str]:
        """Resolve a single import statement to source file."""
        result = {}
        
        # Python
        py_match = re.search(r'from\s+([\w.]+)\s+import\s+(.+)', import_stmt)
        if py_match:
            module = py_match.group(1)
            imports = [s.strip() for s in py_match.group(2).split(',')]
            
            potential_paths = [
                f"{module.replace('.', '/')}.py",
                f"{module.replace('.', '/')}/__init__.py"
            ]
            
            for path in potential_paths:
                if path in self.repo_files:
                    for sym in imports:
                        result[sym.split(' as ')[-1].strip()] = path
                    break
        
        # JavaScript/TypeScript
        js_match = re.search(r'import\s+\{([^}]+)\}\s+from\s+["\']([^"\']+)["\']', import_stmt)
        if js_match:
            imports = [s.strip() for s in js_match.group(1).split(',')]
            path = js_match.group(2)
            
            resolved_path = self._resolve_relative_path(current_file, path)
            if resolved_path and resolved_path in self.repo_files:
                for sym in imports:
                    result[sym.split(' as ')[-1].strip()] = resolved_path
        
        js_default = re.search(r'import\s+(\w+)\s+from\s+["\']([^"\']+)["\']', import_stmt)
        if js_default:
            symbol = js_default.group(1)
            path = js_default.group(2)
            resolved_path = self._resolve_relative_path(current_file, path)
            if resolved_path and resolved_path in self.repo_files:
                result[symbol] = resolved_path
        
        # C/C++ includes
        cpp_match = re.search(r'#include\s+["\']([^"\']+)["\']', import_stmt)
        if cpp_match:
            include_file = cpp_match.group(1)
            resolved_path = self._resolve_relative_path(current_file, include_file)
            if resolved_path and resolved_path in self.repo_files:
                # For C/C++, we don't know specific symbols, so return empty
                pass
        
        return result
    
    def _resolve_relative_path(self, current_file: str, import_path: str) -> Optional[str]:
        """Resolve relative import path to actual file path."""
        if not import_path.startswith('.'):
            return None
        
        current_dir = os.path.dirname(current_file)
        resolved = os.path.normpath(os.path.join(current_dir, import_path))
        
        # Try with different extensions
        extensions = ['.js', '.jsx', '.ts', '.tsx', '/index.js', '/index.ts', '.h', '.hpp']
        for ext in extensions:
            candidate = resolved + ext
            if candidate in self.repo_files:
                return candidate
        
        # Try exact match
        if resolved in self.repo_files:
            return resolved
        
        return None
    
    def resolve_call(self, filepath: str, symbol: str) -> Optional[str]:
        """Resolve a symbol call to its source file."""
        return self.import_map.get(filepath, {}).get(symbol)

# --- 4. Vector Embedding System ---

class VectorEmbeddingStore:
    """Stores and searches function embeddings using FAISS."""
    
    def __init__(self, dimension=1536):
        self.dimension = dimension
        self.index = None
        self.node_ids = []
        self.node_metadata = {}
        
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dimension)
    
    async def build_index(self, graph: Dict[str, Dict]):
        """Build FAISS index from graph nodes."""
        if not HAS_FAISS:
            return
        
        if not graph:
            return
        
        texts = []
        node_ids = []
        
        for node_id, data in graph.items():
            code_preview = data['code'][:500]
            text = f"{node_id} {data['file']} {code_preview}"
            texts.append(text)
            node_ids.append(node_id)
            self.node_metadata[node_id] = data
        
        if not texts:
            return
        
        embeddings = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i+EMBEDDING_BATCH_SIZE]
            batch_emb = await get_embeddings_batch(batch)
            embeddings.extend(batch_emb)
            print(f"      Embedded {min(i+EMBEDDING_BATCH_SIZE, len(texts))}/{len(texts)} nodes", flush=True)
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            self.node_ids = node_ids
    
    async def search(self, query: str, k: int = TOP_K_SEEDS) -> List[str]:
        """Search for top-k most relevant nodes."""
        if not HAS_FAISS or self.index is None or self.index.ntotal == 0:
            return []
        
        query_emb = await get_embeddings_batch([query])
        if not query_emb:
            return []
        
        query_array = np.array(query_emb, dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        k = min(k, len(self.node_ids))
        if k == 0:
            return []
        
        distances, indices = self.index.search(query_array, k)
        result_ids = [self.node_ids[idx] for idx in indices[0] if idx < len(self.node_ids)]
        return result_ids
    
    def save(self, filepath: str):
        """Save index to disk."""
        if HAS_FAISS and self.index and self.index.ntotal > 0:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            faiss.write_index(self.index, filepath)
    
            with open(filepath + '.meta', 'w') as f:
                json.dump({
                    'node_ids': self.node_ids,
                    'node_metadata': self.node_metadata
                }, f)
        else:
            raise RuntimeError(
                "Cannot save index: FAISS unavailable or index is empty."
            )

            
    def load(self, filepath: str):
        """Load index from disk."""
        if HAS_FAISS and os.path.exists(filepath):
            self.index = faiss.read_index(filepath)
            with open(filepath + '.meta', 'r') as f:
                data = json.load(f)
                self.node_ids = data['node_ids']
                self.node_metadata = data['node_metadata']

# --- Graph Building ---

async def build_single_repo_graph(repo_name: str, files_data: Dict[str, Dict]) -> Tuple[Dict, 'ImportResolver']:
    """Build graph with Tree-sitter parsing and import resolution."""
    parser = TreeSitterParser()
    
    parsed_data = {}
    
    print(f"   🔍 Parsing {len(files_data)} files in [{repo_name}]...")
    for filename, data in files_data.items():
        result = parser.parse(filename, data['content'])
        parsed_data[filename] = result
    
    resolver = ImportResolver(files_data)
    resolver.build_maps(parsed_data)
    
    symbol_registry = defaultdict(list)
    file_globals = {}
    
    for filename, result in parsed_data.items():
        file_globals[filename] = result['globals']
        
        for node in result['nodes']:
            name = node['name']
            symbol_registry[name].append({
                "file": filename,
                "type": node['type'],
                "code": node['code'],
                "calls": node['calls'],
                "api_route": node.get('api_route'),
                "api_outbound": node.get('api_outbound', [])
            })
    
    graph = {}
    defined_symbols = set(symbol_registry.keys())
    
    for sym_name, implementations in symbol_registry.items():
        for impl in implementations:
            node_id = f"{impl['file']}::{sym_name}"
            
            valid_deps = []
            
            for called_func in impl['calls']:
                source_file = resolver.resolve_call(impl['file'], called_func)
                
                if source_file:
                    candidates = symbol_registry.get(called_func, [])
                    for candidate in candidates:
                        if candidate['file'] == source_file:
                            target_id = f"{candidate['file']}::{called_func}"
                            if target_id != node_id:
                                valid_deps.append(target_id)
                            break
                else:
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
                "dependencies": list(set(valid_deps)),
                "api_route": impl.get('api_route'),
                "api_outbound": impl.get('api_outbound', []),
                "cross_repo_deps": []
            }
    
    print(f"   ✅ Built graph for [{repo_name}]: {len(graph)} nodes")
    return graph, resolver

def link_cross_repo_dependencies(multi_graph: Dict[str, Dict]) -> Dict[str, Dict]:
    """Link API calls across repositories."""
    
    route_map = defaultdict(list)
    
    for repo_name, graph in multi_graph.items():
        for node_id, data in graph.items():
            route = data.get('api_route')
            if route:
                normalized = normalize_route(route)
                route_map[normalized].append((repo_name, node_id))
    
    count_links = 0
    for repo_name, graph in multi_graph.items():
        for node_id, data in graph.items():
            outbound_routes = data.get('api_outbound', [])
            
            for route in outbound_routes:
                normalized = normalize_route(route)
                
                if normalized in route_map:
                    targets = route_map[normalized]
                    for target_repo, target_node_id in targets:
                        if target_repo == repo_name and target_node_id == node_id:
                            continue
                        
                        cross_ref = {
                            'repo': target_repo,
                            'node_id': target_node_id,
                            'route': route
                        }
                        
                        if cross_ref not in data['cross_repo_deps']:
                            data['cross_repo_deps'].append(cross_ref)
                            count_links += 1
    
    if count_links > 0:
        print(f"🌍 Linking cross-repo API dependencies...")
        print(f"   🔗 Established {count_links} cross-repo API connections")
    
    return multi_graph

async def build_multi_symbol_graph(multi_repo_data: Dict[str, Dict], summarizer: ProjectSummarizer) -> Tuple[Dict, Dict]:
    """Build graphs for all repos with embeddings and summaries."""
    print("\n🕵️ Building symbol graphs with enhanced parsing...")
    
    multi_graph = {}
    resolvers = {}
    
    for repo_name, files_data in multi_repo_data.items():
        graph, resolver = await build_single_repo_graph(repo_name, files_data)
        
        # Project Summarization: Summarize each file
        print(f"   📝 Summarizing {len(files_data)} files in [{repo_name}]...")
        for filename, data in files_data.items():
            nodes = graph.get(filename, {}).get('nodes', []) # This is wrong, graph is node_id based
            # Let's fix this: group nodes by file
            file_nodes = [nd for nid, nd in graph.items() if nd['file'] == filename]
            # nodes in build_single_repo_graph are internal, let's just pass some context
            await summarizer.summarize_file(filename, data['content'], [{"name": k.split("::")[-1]} for k in graph.keys() if graph[k]['file'] == filename])
            
        multi_graph[repo_name] = graph
        resolvers[repo_name] = resolver
    
    multi_graph = link_cross_repo_dependencies(multi_graph)
    
    # Generate project level overview
    await summarizer.generate_project_overview(multi_graph)
    
    vector_stores = {}
    
    # Only build embeddings if we have nodes
    has_nodes = any(len(graph) > 0 for graph in multi_graph.values())
    
    if has_nodes and HAS_FAISS:
        print("   🧬 Building vector embeddings...")
        for repo_name, graph in multi_graph.items():
            if graph:  # Only if graph has nodes
                store = VectorEmbeddingStore()
                await store.build_index(graph)
                vector_stores[repo_name] = store
    
    with open(GRAPH_FILE, 'w') as f:
        json.dump(multi_graph, f, indent=2)
    
    return multi_graph, vector_stores

# --- Enhanced Selector ---

def _extract_specific_targets(query: str, hints: List[str], active_graph: Dict) -> Set[str]:
    """Extract matching node IDs based on hints from reframer."""
    targets = set()
    node_keys = list(active_graph.keys())
    
    for hint in hints:
        hint_lower = hint.lower()
        # Direct match
        if hint in active_graph:
            targets.add(hint)
            continue
            
        # Partial match
        for key in node_keys:
            if hint_lower in key.lower():
                targets.add(key)
                
    return targets

def _backward_traverse(
    seed_nodes: Set[str],
    active_graph: Dict,
    max_depth: int = BACKWARD_TRAVERSAL_DEPTH
) -> Set[str]:
    """
    Traverse from seeds to find all functions they call (callees).
    This builds the call chain: func1 calls func2, func2 calls func3.
    """
    selected = set(seed_nodes)
    queue = list(seed_nodes)
    depth = 0
    
    while queue and depth < max_depth:
        next_queue = []
        for node_id in queue:
            node = active_graph.get(node_id)
            if not node:
                continue
                
            for dep_id in node.get('dependencies', []):
                if dep_id in active_graph and dep_id not in selected:
                    selected.add(dep_id)
                    next_queue.append(dep_id)
        queue = next_queue
        depth += 1
    return selected

async def selector_agent_enhanced(
    target_repo: str,
    technical_query: str,
    multi_graph: Dict[str, Dict],
    vector_stores: Dict[str, VectorEmbeddingStore],
    query_type: str = "FUNCTIONAL_AREA",
    hints: List[str] = None,
    summarizer: ProjectSummarizer = None
) -> List[str]:
    """Enhanced selector using vector search, graph traversal, and summaries."""
    print(f"🗂️ Selector: Strategy [{query_type}] in [{target_repo}]...", flush=True)
    
    if target_repo == "ALL" or target_repo not in multi_graph:
        active_graph = {}
        active_stores = {}
        for r_name, g_data in multi_graph.items():
            for node_id, node_data in g_data.items():
                prefixed_id = f"{r_name}::{node_id}"
                active_graph[prefixed_id] = node_data
            active_stores[r_name] = vector_stores.get(r_name)
    else:
        active_graph = {k: v for k, v in multi_graph[target_repo].items()}
        active_stores = {target_repo: vector_stores.get(target_repo)}
    
    if not active_graph and query_type != "HIGH_LEVEL":
        return []
    
    selected_nodes = set()
    extra_context = []

    # Strategy 1: HIGH_LEVEL - Include Project & File Summaries
    if query_type == "HIGH_LEVEL" and summarizer:
        extra_context.append(f"=== PROJECT ARCHITECTURAL OVERVIEW ===\n\n{summarizer.project_summary}")
        # Also include top file summaries
        file_sum_str = "\n".join([f"- {f}: {s}" for f, s in list(summarizer.file_summaries.items())[:20]])
        extra_context.append(f"=== FILE SUMMARIES ===\n\n{file_sum_str}")
        
    # Strategy 2: SPECIFIC - Backward traversal from hints
    if (query_type == "SPECIFIC" or hints) and active_graph:
        specific_seeds = _extract_specific_targets(technical_query, hints or [], active_graph)
        if specific_seeds:
            print(f"   � Target Seeds: {list(specific_seeds)[:3]}")
            selected_nodes.update(_backward_traverse(specific_seeds, active_graph))

    # Strategy 3: Vector Fallback / FUNCTIONAL_AREA
    if not selected_nodes and active_graph:
        seed_nodes = set()
        for repo, store in active_stores.items():
            if store and HAS_FAISS:
                search_results = await store.search(technical_query, k=TOP_K_SEEDS)
                for result_id in search_results:
                    full_id = f"{repo}::{result_id}" if target_repo == "ALL" else result_id
                    if full_id in active_graph:
                        seed_nodes.add(full_id)
        
        if not seed_nodes:
            seed_nodes = await llm_seed_selection(technical_query, active_graph)
            
        selected_nodes.update(_backward_traverse(seed_nodes, active_graph, max_depth=MAX_RECURSION_DEPTH))

    context_strings = build_context_with_budget(selected_nodes, active_graph, target_repo)
    return extra_context + context_strings

async def llm_seed_selection(query: str, active_graph: Dict) -> Set[str]:
    """Fallback LLM-based seed selection."""
    keys = list(active_graph.keys())[:800]
    
    menu = []
    for node_id in keys:
        data = active_graph[node_id]
        menu.append(f"ID: {node_id} | Type: {data['type']}")
    
    menu_str = "\n".join(menu)
    
    prompt = f"""
You are a Code Navigator.
Query: "{query}"

Available Nodes:
{menu_str}

TASK: Select 2-5 STARTING NODES (IDs) most relevant to the query.
Return JSON: {{"seed_nodes": ["node_id_1", "node_id_2"]}}
"""
    
    try:
        res = await safe_chat_completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        seed_nodes = json.loads(res.choices[0].message.content).get('seed_nodes', [])
        return set(seed_nodes)
    except:
        return set()

def build_context_with_budget(
    selected_nodes: Set[str],
    active_graph: Dict,
    target_repo: str
) -> List[str]:
    """Build context strings with token budgeting."""
    
    files_context = defaultdict(lambda: {"globals": "", "functions": []})

    for node_id in selected_nodes:
        if node_id not in active_graph:
            continue
        
        node = active_graph[node_id]
        
        if target_repo == "ALL" and "::" in node_id:
            parts = node_id.split("::")
            repo_name = parts[0]
            display_name = f"[{repo_name}] {node['file']}"
        else:
            display_name = node['file']
        
        # Globals now contain MUCH more info!
        files_context[display_name]["globals"] = node['globals']
        files_context[display_name]["functions"].append(node['code'])

    context_blocks = []

    for fname, data in files_context.items():
        block = f"=== FILE: {fname} ===\n\n"
        
        # Show globals PROMINENTLY at the top
        if data['globals']:
            block += data['globals']
            block += "\n" + "="*60 + "\n"
            block += "# FUNCTION DEFINITIONS BELOW\n"
            block += "="*60 + "\n\n"
        
        block += "\n\n".join(data['functions'])
        context_blocks.append(block)

    budgeted_context = truncate_to_token_budget(context_blocks, MAX_CONTEXT_TOKENS)

    return budgeted_context

# --- Reframer ---

async def reframer_agent(user_query: str, chat_history: List[Dict], available_repos: List[str]) -> Dict[str, Any]:
    """Detect target repo, query type, and architectural hints."""
    print("🧠 Reframer: Analyzing query...", flush=True)
    
    history_text = ""
    for turn in chat_history[-3:]:
        history_text += f"{turn['role'].upper()}: {turn['content']}\n"
    
    repo_list_str = ", ".join(available_repos)
    
    prompt = f"""
You are a Technical Assistant managing multiple repositories.
Available Repositories: [{repo_list_str}]

Conversation History:
{history_text}

Current Query: "{user_query}"

TASK:
1. Determine the TARGET_REPO (name or 'ALL').
2. Classify QUERY_TYPE as one of:
   - 'SPECIFIC': User asks about a specific function, class, or bug in a known terminal node.
   - 'FUNCTIONAL_AREA': User asks about a feature, module, or broader capability.
   - 'HIGH_LEVEL': User asks about architecture, overview, or "how things work" generally.
3. Extract HINTS: List any filenames, function names, or class names mentioned or implied.
4. Rewrite the QUERY for better retrieval.

OUTPUT FORMAT:
TARGET_REPO: <repo>
QUERY_TYPE: <type>
HINTS: ["hint1", "hint2"]
QUERY: <rewritten_query>
"""
    
    res = await safe_chat_completion(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    content = res.choices[0].message.content.strip()
    
    result = {
        "target_repo": "ALL",
        "query_type": "FUNCTIONAL_AREA",
        "hints": [],
        "query": user_query
    }
    
    match_repo = re.search(r"TARGET_REPO:\s*(.+)", content)
    if match_repo:
        result["target_repo"] = match_repo.group(1).strip()
    
    match_type = re.search(r"QUERY_TYPE:\s*(.+)", content)
    if match_type:
        result["query_type"] = match_type.group(1).strip()
        
    match_hints = re.search(r"HINTS:\s*(\[.+\])", content)
    if match_hints:
        try:
            result["hints"] = json.loads(match_hints.group(1))
        except:
            pass
    
    match_query = re.search(r"QUERY:\s*(.+)", content, re.DOTALL)
    if match_query:
        result["query"] = match_query.group(1).strip()
    
    print(f"   ↳ Target: [{result['target_repo']}] | Type: {result['query_type']} | Hints: {result['hints']}")
    return result

# --- Answering Agent ---

async def answering_agent(user_query: str, context_strings: List[str]) -> str:
    """Generate answer with streaming output."""
    print("📝 Answering Agent: Generating response...", flush=True)
    
    full_context = "\n".join(context_strings)
    context_tokens = count_tokens(full_context)
    print(f"   📊 Context size: {context_tokens} tokens")
    
    messages = [
        {"role": "system", "content": "You are a senior developer. Answer based strictly on the provided Code Context. Be specific and cite file names when relevant."},
        {"role": "user", "content": f"Query: {user_query}\n\nCode Context:\n{full_context}"}
    ]
    
    try:
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True
        )
        
        full_response = ""
        print("\n" + "="*60)
        print("✅ ANSWER:")
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content
        
        print("\n" + "="*60)
        return full_response
        
    except Exception as e:
        print(f"\n❌ Streaming error: {e}")
        res = await safe_chat_completion(model=MODEL_NAME, messages=messages)
        return res.choices[0].message.content

# --- Cache Management ---

def save_cache(multi_graph: Dict, vector_stores: Dict, repo_hashes: Dict, summarizer: ProjectSummarizer):
    """Save graph, embeddings, and summaries to cache."""
    print("💾 Saving cache...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_data = {
        'timestamp': time.time(),
        'repo_hashes': repo_hashes,
        'graph': multi_graph
    }

    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)

    # Save vector indices
    for repo_name, store in vector_stores.items():
        index_path = os.path.join(CACHE_DIR, f"{repo_name}.faiss")
        store.save(index_path)
    
    # Save summaries
    summarizer.save(os.path.join(CACHE_DIR, "summaries.json"))

    print(f"   ✅ Cache saved to {CACHE_DIR}")

def load_cache(current_hashes: Dict, summarizer: ProjectSummarizer) -> Optional[Tuple[Dict, Dict]]:
    """Load cache if valid."""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
        
        cached_hashes = cache_data.get('repo_hashes', {})
        if cached_hashes != current_hashes:
            print("⚠️ Repository changes detected, invalidating cache")
            return None
        
        print("✅ Loading from cache...")
        multi_graph = cache_data['graph']
        
        vector_stores = {}
        for repo_name in multi_graph.keys():
            store = VectorEmbeddingStore()
            index_path = os.path.join(CACHE_DIR, f"{repo_name}.faiss")
            if os.path.exists(index_path):
                store.load(index_path)
                vector_stores[repo_name] = store
        
        # Load summaries
        summarizer.load(os.path.join(CACHE_DIR, "summaries.json"))
        
        return multi_graph, vector_stores
    except Exception as e:
        print(f"⚠️ Cache load error: {e}")
        return None

# --- Main ---

async def main():
    chat_history = []
    init_cache_dir()
    summarizer = ProjectSummarizer()

    try:
        print("🔗 === GITHUB SOURCE CONFIGURATION === 🔗")
        gh_input = input("\n🐙 Enter GitHub Repos (comma-separated): ")
        if not gh_input.strip():
            return
        
        multi_repo_data, repo_hashes = await ingest_sources(gh_input)
        if not multi_repo_data:
            return
        
        cached = load_cache(repo_hashes, summarizer)
        
        if cached:
            multi_graph, vector_stores = cached
        else:
            multi_graph, vector_stores = await build_multi_symbol_graph(multi_repo_data, summarizer)
            save_cache(multi_graph, vector_stores, repo_hashes, summarizer)
        
        available_repos = list(multi_graph.keys())
        
        # Check if we have any nodes
        total_nodes = sum(len(graph) for graph in multi_graph.values())
        
        if total_nodes == 0:
            print("\n⚠️ Warning: No code symbols were extracted from the repository.")
            print("\n   You can still ask questions, but responses may be limited.")
        
        print(f"\n✅ System ready! Found {total_nodes} symbols across {len(available_repos)} repo(s).")
        print("   Ask questions about your code.")
        
        while True:
            query = input("\n" + "-"*40 + "\n(Type 'exit' to quit) Question: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            reframer_res = await reframer_agent(query, chat_history, available_repos)
            
            context_strings = await selector_agent_enhanced(
                reframer_res["target_repo"],
                reframer_res["query"],
                multi_graph,
                vector_stores,
                query_type=reframer_res["query_type"],
                hints=reframer_res["hints"],
                summarizer=summarizer
            )
            
            if not context_strings:
                print("   ⚠️ No relevant code found.")
                continue
            
            answer = await answering_agent(query, context_strings)
            
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        perform_cleanup()

if __name__ == "__main__":
    asyncio.run(main())