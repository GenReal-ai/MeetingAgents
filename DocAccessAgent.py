import os
import json
import time
import wave
import threading
import pyaudio
import difflib
from typing import List, Dict, Union, Tuple
from dotenv import load_dotenv
from colorama import Fore, Style, init
from bs4 import BeautifulSoup, Tag

# APIs
from atlassian import Confluence
from openai import OpenAI

# --- CONFIGURATION ---
load_dotenv()
init(autoreset=True)

# Audio Config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
AUDIO_FILENAME = "command.wav"

# Confluence Config
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_USER = os.getenv("CONFLUENCE_USER")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

client = OpenAI()

# ==========================================
# 1. THE SAFETY VALIDATOR
# ==========================================
class SafetyValidator:
    """
    The Safety Net. Ensures the AI doesn't break the page.
    """
    @staticmethod
    def validate(original_html: str, new_html: str, intent_summary: str) -> bool:
        print(f"{Fore.YELLOW}🛡️  [Validator] Checking safety...{Style.RESET_ALL}")
        
        # Rule 1: Empty Check
        if not new_html or len(new_html.strip()) < 5:
            print(f"{Fore.RED}❌ REJECTED: Result was empty.{Style.RESET_ALL}")
            return False

        # Rule 2: Macro Preservation (Critical for Confluence)
        # We count <ac:structured-macro> tags.
        orig_macros = original_html.count("ac:structured-macro")
        new_macros = new_html.count("ac:structured-macro")
        
        # If macros disappeared and the user didn't explicitly ask to delete things
        if orig_macros > new_macros and "delete" not in intent_summary.lower():
            print(f"{Fore.RED}❌ REJECTED: Macros lost! (Old: {orig_macros}, New: {new_macros}){Style.RESET_ALL}")
            return False

        # Rule 3: Sanity Check on Size Variance
        # If the section shrank by >80% without 'delete' intent, it's suspicious
        if len(new_html) < len(original_html) * 0.2 and "delete" not in intent_summary.lower():
            print(f"{Fore.RED}❌ REJECTED: Significant content loss detected.{Style.RESET_ALL}")
            return False

        return True

# ==========================================
# 2. DOCUMENT SURGEON (DOM TOOLS)
# ==========================================
class DocumentSurgeon:
    """
    Handles parsing Confluence Storage Format (XHTML) and performing
    surgical operations using BeautifulSoup.
    """
    def __init__(self, html_content):
        # 'xml' parser is required to preserve specific Confluence tags
        self.soup = BeautifulSoup(html_content, 'xml') 

    def get_annotated_structure(self) -> str:
        """
        Returns a simplified map of the page for the Brain to analyze.
        """
        map_text = ""
        headers = self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for idx, tag in enumerate(headers):
            # Preview the first 50 chars of content under the header
            preview = ""
            for sibling in tag.next_siblings:
                if isinstance(sibling, Tag) and sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']: break
                if hasattr(sibling, 'get_text'): preview += sibling.get_text()[:50].replace("\n", " ")
            
            map_text += f"[ID: {idx}] <{tag.name}> {tag.get_text().strip()} (Content: {preview}...)\n"
        
        return map_text if map_text else "No Headers Found."

    def get_section_html(self, section_id: int) -> str:
        """Fetches the RAW HTML for a specific section ID."""
        headers = self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if section_id >= len(headers) or section_id < 0: return ""
        
        target_header = headers[section_id]
        start_level = int(target_header.name[1])
        
        html_output = [str(target_header)]
        
        # Walk siblings
        for sibling in target_header.next_siblings:
            if isinstance(sibling, Tag) and sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if int(sibling.name[1]) <= start_level: break # Stop at equal or higher header
            html_output.append(str(sibling))
            
        return "".join(html_output)

    def replace_section(self, section_id: int, new_html: str):
        """Swaps the old section with new HTML."""
        headers = self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if section_id >= len(headers): return False

        target_header = headers[section_id]
        start_level = int(target_header.name[1])
        
        # 1. Identify nodes to remove
        nodes_to_remove = [target_header]
        insertion_point = None
        
        for sibling in target_header.next_siblings:
            if isinstance(sibling, Tag) and sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if int(sibling.name[1]) <= start_level: 
                    insertion_point = sibling # Found the boundary
                    break
            nodes_to_remove.append(sibling)
        
        # 2. Extract (remove) old nodes
        for node in nodes_to_remove:
            node.extract()

        # 3. Insert new nodes
        # If we found an insertion point (the next header), insert before it
        # If not (end of doc), append to body
        new_soup = BeautifulSoup(new_html, 'xml')
        new_nodes = list(new_soup.children)

        if insertion_point:
            for node in new_nodes:
                insertion_point.insert_before(node)
        else:
            parent = self.soup.body if self.soup.body else self.soup
            for node in new_nodes:
                parent.append(node)
            
        return True

    def get_full_html(self):
        return str(self.soup)

# ==========================================
# 3. INTELLIGENT AGENTS
# ==========================================
class AnalystAgent:
    """Determines Intent."""
    def analyze(self, transcript: str):
        print(f"\n{Fore.CYAN}🕵️  [Analyst] Hearing: '{transcript}'{Style.RESET_ALL}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a Confluence Editor Assistant. 
                Extract intent from the user command.
                Output JSON: { "requires_change": bool, "summary": str, "specific_instructions": str }"""},
                {"role": "user", "content": transcript}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

class SpotlightBrain:
    """The Logic Engine."""
    
    def identify_scope(self, intent: dict, map_text: str) -> List[int]:
        """Returns List of Section IDs to modify."""
        print(f"{Fore.MAGENTA}🧠 [Brain] Scoping Document...{Style.RESET_ALL}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a dependency analyzer. Given a document map, identify the IDs of sections that must change based on the user intent. Return JSON: {'section_ids': [int]}"},
                {"role": "user", "content": f"Intent: {intent['specific_instructions']}\n\nMap:\n{map_text}"}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content).get("section_ids", [])

    def process_section_logic(self, section_html: str, intent: dict) -> str:
        """Rewrites the specific section based on logic."""
        print(f"{Fore.MAGENTA}🧠 [Brain] Rewriting Logic (Dates/Math/Text)...{Style.RESET_ALL}")
        
        system_prompt = """
        You are a Confluence XHTML Expert. 
        1. Read the provided HTML section.
        2. Apply the user's specific instructions (e.g., increment dates, change text).
        3. Return the FULL, VALID XHTML for this section.
        4. CRITICAL: Preserve all <ac:structured-macro> tags exactly as they are.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Instructions: {intent['specific_instructions']}\n\nHTML:\n{section_html}"}
            ]
        )
        return response.choices[0].message.content

# ==========================================
# 4. MAIN ORCHESTRATOR
# ==========================================
class ConfluenceAgentSystem:
    def __init__(self, page_id):
        self.page_id = page_id
        self.confluence = Confluence(
            url=CONFLUENCE_URL, username=CONFLUENCE_USER, password=CONFLUENCE_API_TOKEN, cloud=True
        )
        self.analyst = AnalystAgent()
        self.brain = SpotlightBrain()
        self.validator = SafetyValidator()

    def process_command(self, transcript_text):
        # 1. Analyze
        intent = self.analyst.analyze(transcript_text)
        if not intent["requires_change"]:
            print("No changes requested.")
            return

        # 2. Fetch Page & Map
        print(f"{Fore.BLUE}📥 Fetching Page Data...{Style.RESET_ALL}")
        try:
            page = self.confluence.get_page_by_id(self.page_id, expand='body.storage,version')
            full_html = page['body']['storage']['value']
            title = page['title']
            version = page['version']['number']
        except Exception as e:
            print(f"{Fore.RED}Error fetching page: {e}{Style.RESET_ALL}")
            return

        surgeon = DocumentSurgeon(full_html)
        doc_map = surgeon.get_annotated_structure()

        # 3. Scope (Spotlight)
        target_ids = self.brain.identify_scope(intent, doc_map)
        if not target_ids:
            print(f"{Fore.YELLOW}⚠️ Brain could not match intent to any section.{Style.RESET_ALL}")
            return

        print(f"{Fore.CYAN}🎯 Spotlight on Sections: {target_ids}{Style.RESET_ALL}")

        # 4. Processing Loop
        dirty = False
        for sid in target_ids:
            # A. Get Context
            original_section = surgeon.get_section_html(sid)
            
            # B. Generate New Content
            new_section = self.brain.process_section_logic(original_section, intent)
            
            # C. Validate
            is_valid = self.validator.validate(original_section, new_section, intent["summary"])
            
            if is_valid:
                # D. Apply to Local DOM
                success = surgeon.replace_section(sid, new_section)
                if success:
                    print(f"{Fore.GREEN}✅ Section {sid} Updated locally.{Style.RESET_ALL}")
                    dirty = True
            else:
                print(f"{Fore.RED}⛔ Section {sid} Skipped (Validation Failed).{Style.RESET_ALL}")

        # 5. Final Push
        if dirty:
            print(f"{Fore.BLUE}📤 Uploading changes to Confluence...{Style.RESET_ALL}")
            final_html = surgeon.get_full_html()
            self.confluence.update_page(self.page_id, title=title, body=final_html)
            print(f"{Fore.GREEN}🚀 SUCCESS: Page updated.{Style.RESET_ALL}")
        else:
            print("No changes were applied.")

# ==========================================
# 5. AUDIO RECORDER UTILITY
# ==========================================
class AudioRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.recording = False

    def record_thread(self):
        stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while self.recording:
            data = stream.read(CHUNK)
            self.frames.append(data)
        stream.stop_stream()
        stream.close()

    def start(self):
        self.frames = []
        self.recording = True
        self.thread = threading.Thread(target=self.record_thread)
        self.thread.start()
        print(f"{Fore.RED}🔴 Recording... (Press ENTER to stop){Style.RESET_ALL}")

    def stop(self):
        self.recording = False
        self.thread.join()
        wf = wave.open(AUDIO_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"{Fore.GREEN}⏹️ Recording saved.{Style.RESET_ALL}")

# ==========================================
# 6. RUNNER
# ==========================================
if __name__ == "__main__":
    print(f"{Fore.YELLOW}=== CONFLUENCE AI AGENT 2.0 (SPOTLIGHT & VALIDATOR) ==={Style.RESET_ALL}")
    
    # Get ID
    page_id = input("Enter Confluence Page ID: ").strip()
    
    system = ConfluenceAgentSystem(page_id)
    recorder = AudioRecorder()

    while True:
        try:
            input(f"\n{Fore.WHITE}Press ENTER to start recording...{Style.RESET_ALL}")
            recorder.start()
            input() # Wait for Enter
            recorder.stop()

            # Transcribe
            print("📝 Transcribing...")
            with open(AUDIO_FILENAME, "rb") as f:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=f).text
            
            # Execute Pipeline
            system.process_command(transcript)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"{Fore.RED}System Error: {e}{Style.RESET_ALL}")