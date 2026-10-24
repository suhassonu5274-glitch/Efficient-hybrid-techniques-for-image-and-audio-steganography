"""
gui.py - simple Tkinter front-end (image + audio tabs).
Run:  python gui.py
"""
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import audio_stego as aud
import image_stego as img
from stego_core import psnr, snr_db


class StegoTab(ttk.Frame):
    def __init__(self, master, kind):
        super().__init__(master, padding=10)
        self.kind = kind
        self.ext = ".png" if kind == "image" else ".wav"
        ftypes = [("PNG/JPG/BMP", "*.png *.jpg *.jpeg *.bmp")] if kind == "image" else [("WAV", "*.wav")]
        self.ftypes = ftypes
        self.cover = tk.StringVar()
        self.output = tk.StringVar()
        self.stego = tk.StringVar()
        self.key = tk.StringVar()
        self.mode = tk.StringVar(value="lsb")

        r = 0
        self._file_row("Cover file", self.cover, r, open_=True); r += 1
        self._file_row(f"Save stego as ({self.ext})", self.output, r, open_=False); r += 1
        ttk.Label(self, text="Mode").grid(row=r, column=0, sticky="w")
        ttk.Combobox(self, textvariable=self.mode, values=["lsb", "dct"],
                     state="readonly", width=8).grid(row=r, column=1, sticky="w"); r += 1
        ttk.Label(self, text="Secret key").grid(row=r, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.key, show="*", width=30).grid(row=r, column=1, sticky="w"); r += 1
        ttk.Label(self, text="Secret message").grid(row=r, column=0, sticky="nw")
        self.msg = scrolledtext.ScrolledText(self, width=60, height=5)
        self.msg.grid(row=r, column=1, columnspan=2, pady=4); r += 1
        ttk.Button(self, text="Embed", command=self.do_embed).grid(row=r, column=1, sticky="w"); r += 1
        ttk.Separator(self).grid(row=r, column=0, columnspan=3, sticky="ew", pady=8); r += 1
        self._file_row("Stego file", self.stego, r, open_=True); r += 1
        ttk.Button(self, text="Extract", command=self.do_extract).grid(row=r, column=1, sticky="w"); r += 1
        self.log = scrolledtext.ScrolledText(self, width=60, height=6, state="disabled")
        self.log.grid(row=r, column=0, columnspan=3, pady=6)

    def _file_row(self, label, var, row, open_):
        ttk.Label(self, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(self, textvariable=var, width=50).grid(row=row, column=1)
        def browse():
            if open_:
                p = filedialog.askopenfilename(filetypes=self.ftypes)
            else:
                p = filedialog.asksaveasfilename(defaultextension=self.ext)
            if p:
                var.set(p)
        ttk.Button(self, text="Browse", command=browse).grid(row=row, column=2)

    def _say(self, text):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", text)
        self.log.configure(state="disabled")

    def do_embed(self):
        try:
            message = self.msg.get("1.0", "end-1c").encode()
            if not (self.cover.get() and self.output.get() and self.key.get() and message):
                raise ValueError("Fill in cover, output, key and message.")
            if self.kind == "image":
                c, s = img.embed(self.cover.get(), self.output.get(), message, self.key.get(), self.mode.get())
                self._say(f"Embedded. PSNR = {psnr(c, s, 255):.2f} dB\nSaved: {self.output.get()}")
            else:
                c, s = aud.embed(self.cover.get(), self.output.get(), message, self.key.get(), self.mode.get())
                self._say(f"Embedded. SNR = {snr_db(c, s):.2f} dB\nSaved: {self.output.get()}")
            self.stego.set(self.output.get())
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def do_extract(self):
        try:
            if not (self.stego.get() and self.key.get()):
                raise ValueError("Choose a stego file and enter the key.")
            mod = img if self.kind == "image" else aud
            data = mod.extract(self.stego.get(), self.key.get(), self.mode.get())
            self._say("Recovered message:\n\n" + data.decode(errors="replace"))
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    root.title("Hybrid Image & Audio Steganography")
    nb = ttk.Notebook(root)
    nb.add(StegoTab(nb, "image"), text="Image")
    nb.add(StegoTab(nb, "audio"), text="Audio")
    nb.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
