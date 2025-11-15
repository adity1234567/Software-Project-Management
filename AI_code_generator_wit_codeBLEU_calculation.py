import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import google.generativeai as genai
import re
import os
import ast
import math
from collections import Counter
from typing import List, Dict
import threading

# Configure API key
API_KEY = "AIzaSyBsLNLyU8pNyUjnNDyvN_JSubGVV0M9yjU"
genai.configure(api_key=API_KEY)

# Initialize the model
model = genai.GenerativeModel('models/gemini-2.5-flash')

class CodeBLEUEvaluator:
    """CodeBLEU Evaluation Engine"""
    def __init__(self):
        self.weights = {
            'bleu': 0.25,
            'syntax': 0.25,
            'dataflow': 0.25,
            'ngram_match': 0.25
        }
    
    def evaluate(self, reference_code: str, generated_code: str, language: str = 'python') -> Dict:
        ref_clean = self.clean_code(reference_code)
        gen_clean = self.clean_code(generated_code)
        
        bleu_score = self.calculate_bleu(ref_clean, gen_clean)
        syntax_score = self.calculate_syntax_match(reference_code, generated_code, language)
        dataflow_score = self.calculate_dataflow_match(reference_code, generated_code)
        ngram_score = self.calculate_ngram_match(ref_clean, gen_clean)
        
        codebleu_score = (
            self.weights['bleu'] * bleu_score +
            self.weights['syntax'] * syntax_score +
            self.weights['dataflow'] * dataflow_score +
            self.weights['ngram_match'] * ngram_score
        )
        
        return {
            'codebleu_score': round(codebleu_score, 4),
            'bleu_score': round(bleu_score, 4),
            'syntax_match': round(syntax_score, 4),
            'dataflow_match': round(dataflow_score, 4),
            'ngram_match': round(ngram_score, 4),
            'evaluation': self.get_evaluation_level(codebleu_score)
        }
    
    def clean_code(self, code: str) -> str:
        code = re.sub(r'//.*?\n|#.*?\n', '\n', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'\s+', ' ', code)
        return code.strip()
    
    def tokenize(self, code: str) -> List[str]:
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return [t for t in tokens if t.strip()]
    
    def calculate_bleu(self, reference: str, generated: str, max_n: int = 4) -> float:
        ref_tokens = self.tokenize(reference)
        gen_tokens = self.tokenize(generated)
        
        if not gen_tokens:
            return 0.0
        
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = self.get_ngrams(ref_tokens, n)
            gen_ngrams = self.get_ngrams(gen_tokens, n)
            
            if not gen_ngrams:
                precisions.append(0)
                continue
            
            matches = sum((ref_ngrams & gen_ngrams).values())
            total = sum(gen_ngrams.values())
            precisions.append(matches / total if total > 0 else 0)
        
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0
        
        ref_len = len(ref_tokens)
        gen_len = len(gen_tokens)
        bp = 1 if gen_len > ref_len else (math.exp(1 - ref_len / gen_len) if gen_len > 0 else 0)
        
        return bp * geo_mean
    
    def get_ngrams(self, tokens: List[str], n: int) -> Counter:
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return Counter(ngrams)
    
    def calculate_syntax_match(self, reference: str, generated: str, language: str) -> float:
        if language != 'python':
            return self.calculate_keyword_match(reference, generated)
        
        try:
            ref_tree = ast.parse(reference)
            gen_tree = ast.parse(generated)
            
            ref_nodes = self.get_ast_nodes(ref_tree)
            gen_nodes = self.get_ast_nodes(gen_tree)
            
            if not ref_nodes:
                return 0.0
            
            intersection = len(ref_nodes & gen_nodes)
            union = len(ref_nodes | gen_nodes)
            return intersection / union if union > 0 else 0.0
        except:
            return 0.1
    
    def get_ast_nodes(self, tree) -> set:
        return {type(node).__name__ for node in ast.walk(tree)}
    
    def calculate_keyword_match(self, reference: str, generated: str) -> float:
        keywords = ['class', 'public', 'private', 'static', 'void', 'int', 'return', 'if', 'for', 'while']
        ref_lower = reference.lower()
        gen_lower = generated.lower()
        
        matches = sum(1 for kw in keywords if kw in ref_lower and kw in gen_lower)
        total = sum(1 for kw in keywords if kw in ref_lower)
        return matches / total if total > 0 else 0.5
    
    def calculate_dataflow_match(self, reference: str, generated: str) -> float:
        ref_vars = self.extract_variables(reference)
        gen_vars = self.extract_variables(generated)
        
        if not ref_vars:
            return 1.0 if not gen_vars else 0.5
        
        intersection = len(ref_vars & gen_vars)
        union = len(ref_vars | gen_vars)
        return intersection / union if union > 0 else 0.0
    
    def extract_variables(self, code: str) -> set:
        variables = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        keywords = {'if', 'else', 'for', 'while', 'return', 'class', 'def', 'import', 
                   'from', 'print', 'int', 'float', 'str', 'void', 'public', 'private'}
        return set(v for v in variables if v not in keywords)
    
    def calculate_ngram_match(self, reference: str, generated: str) -> float:
        ref_tokens = self.tokenize(reference)
        gen_tokens = self.tokenize(generated)
        
        score_2 = self.ngram_precision(ref_tokens, gen_tokens, 2)
        score_3 = self.ngram_precision(ref_tokens, gen_tokens, 3)
        return (score_2 + score_3) / 2
    
    def ngram_precision(self, ref_tokens: List[str], gen_tokens: List[str], n: int) -> float:
        ref_ngrams = self.get_ngrams(ref_tokens, n)
        gen_ngrams = self.get_ngrams(gen_tokens, n)
        
        if not gen_ngrams:
            return 0.0
        
        matches = sum((ref_ngrams & gen_ngrams).values())
        total = sum(gen_ngrams.values())
        return matches / total if total > 0 else 0.0
    
    def get_evaluation_level(self, score: float) -> str:
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.75:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.4:
            return "Poor"
        return "Very Poor"


class CodeGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Code Generator with CodeBLEU Evaluation")
        self.root.geometry("1000x750")
        self.root.configure(bg="#1e1e1e")
        
        self.generated_code = ""
        self.detected_language = ""
        self.full_code_to_show = ""
        self.evaluator = CodeBLEUEvaluator()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#2d2d2d", height=60)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="🤖 AI Code Generator with Quality Check",
            font=("Arial", 18, "bold"),
            bg="#2d2d2d",
            fg="#61dafb"
        )
        title_label.pack(pady=10)
        
        # Chat Display
        chat_frame = tk.Frame(self.root, bg="#1e1e1e")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#2d2d2d",
            fg="#ffffff",
            insertbackground="#61dafb",
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Tags
        self.chat_display.tag_config("user", foreground="#61dafb", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("bot", foreground="#98c379", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("code", foreground="#e5c07b", font=("Consolas", 9))
        self.chat_display.tag_config("info", foreground="#c678dd", font=("Arial", 9, "italic"))
        self.chat_display.tag_config("score", foreground="#56b6c2", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("clickable", foreground="#61dafb", underline=True)
        
        # Input Frame
        input_frame = tk.Frame(self.root, bg="#1e1e1e")
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.input_box = tk.Text(
            input_frame,
            height=3,
            font=("Arial", 11),
            bg="#2d2d2d",
            fg="#ffffff",
            insertbackground="#61dafb",
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.input_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_box.bind("<Return>", self.handle_enter)
        
        button_frame = tk.Frame(input_frame, bg="#1e1e1e")
        button_frame.pack(side=tk.RIGHT)
        
        self.send_button = tk.Button(
            button_frame,
            text="Generate",
            command=self.send_message,
            bg="#61dafb",
            fg="#1e1e1e",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.send_button.pack()
        
        # Action Buttons
        action_frame = tk.Frame(self.root, bg="#1e1e1e")
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.download_button = tk.Button(
            action_frame,
            text="💾 Download",
            command=self.download_code,
            bg="#98c379",
            fg="#1e1e1e",
            font=("Arial", 10, "bold"),
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.download_button.pack(side=tk.LEFT, padx=5)
        
        self.evaluate_button = tk.Button(
            action_frame,
            text="📊 Evaluate Code",
            command=self.open_evaluation_dialog,
            bg="#c678dd",
            fg="#1e1e1e",
            font=("Arial", 10, "bold"),
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.evaluate_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(
            action_frame,
            text="🗑️ Clear",
            command=self.clear_chat,
            bg="#e06c75",
            fg="#1e1e1e",
            font=("Arial", 10, "bold"),
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor="hand2"
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Welcome
        self.add_bot_message("Welcome! I generate code and evaluate its quality using CodeBLEU metric! 🚀")
    
    def handle_enter(self, event):
        if event.state & 0x1:
            return
        self.send_message()
        return "break"
    
    def add_user_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n")
        self.chat_display.insert(tk.END, "You: ", "user")
        self.chat_display.insert(tk.END, f"{message}\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def add_bot_message(self, message, tag="bot"):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n")
        self.chat_display.insert(tk.END, "Bot: ", tag)
        self.chat_display.insert(tk.END, f"{message}\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def add_code_preview(self, code, language):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n")
        self.chat_display.insert(tk.END, f"📝 Generated Code ({language}):\n", "info")
        
        code_lines = code.split('\n')
        preview_lines = code_lines[:15]
        preview = '\n'.join(preview_lines)
        
        self.chat_display.insert(tk.END, f"{preview}\n", "code")
        
        if len(code_lines) > 15:
            self.full_code_to_show = code
            link_text = f"... ({len(code_lines) - 15} more lines) - Click to show full code\n"
            self.chat_display.insert(tk.END, link_text, "clickable")
            self.chat_display.tag_bind("clickable", "<Button-1>", self.show_full_code)
            self.chat_display.tag_bind("clickable", "<Enter>", lambda e: self.chat_display.config(cursor="hand2"))
            self.chat_display.tag_bind("clickable", "<Leave>", lambda e: self.chat_display.config(cursor=""))
        
        self.chat_display.insert(tk.END, "\n✅ Code generated! Use buttons below to download or evaluate.\n", "info")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def show_full_code(self, event=None):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n--- Full Code ---\n", "info")
        self.chat_display.insert(tk.END, f"{self.full_code_to_show}\n", "code")
        self.chat_display.insert(tk.END, "--- End ---\n", "info")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def detect_language(self, code):
        patterns = {
            'python': [r'def ', r'import ', r'print\('],
            'java': [r'public class', r'System\.out\.'],
            'cpp': [r'#include', r'std::'],
            'javascript': [r'function ', r'const ']
        }
        for lang, pats in patterns.items():
            if any(re.search(p, code, re.IGNORECASE) for p in pats):
                return lang
        return 'txt'
    
    def get_file_extension(self, language):
        ext = {'python': '.py', 'java': '.java', 'cpp': '.cpp', 'javascript': '.js'}
        return ext.get(language, '.txt')
    
    def generate_code(self, query):
        prompt = f"""Generate ONLY executable code. No explanations, no markdown, no comments.
Request: {query}"""
        try:
            response = model.generate_content(prompt)
            code = re.sub(r'^```[\w]*\n', '', response.text)
            code = re.sub(r'\n```$', '', code).strip()
            return code, None
        except Exception as e:
            return None, str(e)
    
    def send_message(self):
        msg = self.input_box.get("1.0", tk.END).strip()
        if not msg:
            return
        
        self.input_box.delete("1.0", tk.END)
        self.add_user_message(msg)
        
        self.send_button.config(state=tk.DISABLED, text="Generating...")
        self.download_button.config(state=tk.DISABLED)
        self.evaluate_button.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.generate_and_display, args=(msg,))
        thread.daemon = True
        thread.start()
    
    def generate_and_display(self, msg):
        code, error = self.generate_code(msg)
        
        if error:
            self.root.after(0, lambda: self.add_bot_message(f"❌ {error}"))
            self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL, text="Generate"))
        else:
            lang = self.detect_language(code)
            self.generated_code = code
            self.detected_language = lang
            
            self.root.after(0, lambda: self.add_code_preview(code, lang))
            self.root.after(0, lambda: self.download_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.evaluate_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL, text="Generate"))
    
    def download_code(self):
        if not self.generated_code:
            messagebox.showwarning("No Code", "Generate code first!")
            return
        
        ext = self.get_file_extension(self.detected_language)
        path = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[(f"{self.detected_language.upper()}", f"*{ext}"), ("All", "*.*")],
            initialfile=f"generated_code{ext}"
        )
        
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(self.generated_code)
                messagebox.showinfo("Success", f"Saved to:\n{path}")
                self.add_bot_message(f"✅ Downloaded: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def open_evaluation_dialog(self):
        if not self.generated_code:
            messagebox.showwarning("No Code", "Generate code first!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("CodeBLEU Evaluation")
        dialog.geometry("600x500")
        dialog.configure(bg="#1e1e1e")
        
        tk.Label(
            dialog,
            text="📊 CodeBLEU Evaluation",
            font=("Arial", 16, "bold"),
            bg="#1e1e1e",
            fg="#61dafb"
        ).pack(pady=10)
        
        tk.Label(
            dialog,
            text="Paste reference/correct code below:",
            font=("Arial", 11),
            bg="#1e1e1e",
            fg="#ffffff"
        ).pack(pady=5)
        
        ref_text = scrolledtext.ScrolledText(
            dialog,
            height=15,
            font=("Consolas", 10),
            bg="#2d2d2d",
            fg="#ffffff"
        )
        ref_text.pack(padx=20, fill=tk.BOTH, expand=True)
        
        result_label = tk.Label(
            dialog,
            text="",
            font=("Arial", 10),
            bg="#1e1e1e",
            fg="#98c379",
            justify=tk.LEFT
        )
        result_label.pack(pady=10)
        
        def evaluate():
            ref = ref_text.get("1.0", tk.END).strip()
            if not ref:
                messagebox.showwarning("Empty", "Enter reference code!")
                return
            
            result = self.evaluator.evaluate(ref, self.generated_code, self.detected_language)
            
            result_text = f"""
CodeBLEU Score: {result['codebleu_score']} / 1.0
├─ BLEU: {result['bleu_score']}
├─ Syntax: {result['syntax_match']}
├─ Dataflow: {result['dataflow_match']}
└─ N-gram: {result['ngram_match']}

Evaluation: {result['evaluation']}
            """
            result_label.config(text=result_text)
            
            self.add_bot_message(f"📊 Evaluation: CodeBLEU={result['codebleu_score']} ({result['evaluation']})")
        
        tk.Button(
            dialog,
            text="Evaluate",
            command=evaluate,
            bg="#c678dd",
            fg="#1e1e1e",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=8
        ).pack(pady=10)
    
    def clear_chat(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        self.generated_code = ""
        self.detected_language = ""
        self.full_code_to_show = ""
        self.download_button.config(state=tk.DISABLED)
        self.evaluate_button.config(state=tk.DISABLED)
        
        self.add_bot_message("Chat cleared! Ready for new request. 🚀")

def main():
    root = tk.Tk()
    app = CodeGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()