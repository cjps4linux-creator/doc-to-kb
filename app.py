import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

class App:
    def __init__(self, root):
        root.title('Document-to-KB')
        root.geometry('1000x700')
        root.resizable(True, True)
        self.root = root
        self.folder = None
        self.files = []
        self.chunks = []
        self.embs = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(frm)
        top.pack(fill=tk.X)
        ttk.Button(top, text='Choose PDF Folder', command=self.choose).pack(side=tk.LEFT)
        ttk.Button(top, text='Index Folder', command=self.index).pack(side=tk.LEFT, padx=2)
        ttk.Label(top, text='Query:').pack(side=tk.LEFT, padx=5)
        self.q = ttk.Entry(top)
        self.q.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(top, text='Search', command=self.search).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text='Clear', command=self.clear).pack(side=tk.LEFT, padx=2)
        mid = ttk.Panedwindow(frm, orient=tk.HORIZONTAL)
        mid.pack(fill=tk.BOTH, expand=True, pady=5)
        left = ttk.Frame(mid)
        self.listbox = tk.Listbox(left)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        mid.add(left, weight=1)
        right = ttk.Frame(mid)
        self.text = tk.Text(right, wrap=tk.WORD)
        self.text.pack(fill=tk.BOTH, expand=True)
        mid.add(right, weight=2)
        ttk.Label(frm, text='Local offline KB').pack(anchor='w')
    
    def choose(self):
        d = filedialog.askdirectory()
        if not d: return
        self.folder = d
        self.refresh()
        self.text.insert(tk.END, f'Selected: {d}\n')
    
    def refresh(self):
        self.listbox.delete(0, tk.END)
        self.files = []
        if not self.folder: return
        for ext in ('*.pdf',):
            for f in sorted(glob.glob(os.path.join(self.folder, ext))):
                self.files.append(f)
                self.listbox.insert(tk.END, os.path.basename(f))
    
    def chunk_text(self, text, size=400):
        words = text.split()
        for i in range(0, len(words), size):
            yield ' '.join(words[i:i+size])
    
    def extract_pdf(self, path):
        t = []
        try:
            r = PdfReader(path)
            for p in r.pages:
                try:
                    t.append(p.extract_text() or '')
                except:
                    pass
        except Exception as e:
            self.text.insert(tk.END, f'  err {os.path.basename(path)}: {e}\n')
        return '\n'.join(t)
    
    def index(self):
        self.refresh()
        self.chunks = []
        for f in self.files:
            self.text.insert(tk.END, f'Indexing {os.path.basename(f)}...\n')
            txt = self.extract_pdf(f)
            for c in self.chunk_text(txt):
                self.chunks.append((f, c))
        if self.chunks:
            embs = self.model.encode([c for _, c in self.chunks], normalize_embeddings=True)
            self.embs = np.array(embs)
        self.text.insert(tk.END, f'Done: {len(self.chunks)} chunks\n')
    
    def search(self):
        q = self.q.get().strip()
        if not q or self.embs is None or len(self.chunks) == 0:
            messagebox.showwarning('KB', 'Index first')
            return
        qe = self.model.encode([q], normalize_embeddings=True)
        sims = self.embs @ qe.T
        idx = np.argsort(-sims.flatten())[:5]
        self.text.insert(tk.END, f'\nQ: {q}\n')
        for i in idx:
            f, c = self.chunks[i]
            self.text.insert(tk.END, f'\n--- {os.path.basename(f)} (sim={sims[i][0]:.3f}) ---\n{c[:400]}...\n')

    def clear(self):
        self.text.delete(1.0, tk.END)

if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
