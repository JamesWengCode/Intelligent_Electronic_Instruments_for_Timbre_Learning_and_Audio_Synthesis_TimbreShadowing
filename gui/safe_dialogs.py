"""Drop-in replacement for tkinter.messagebox that avoids native NSAlert.

On recent macOS, tkinter.messagebox opens a native NSAlert, which macOS wires
up to the private GameControllerUI framework for controller navigation. That
framework has a reproducible crash (SIGTRAP in
GCUINavigationSession.AlertPresentationAssertion dealloc) triggered on a later,
unrelated pass of the Tk event loop after an alert is shown. These dialogs are
plain Toplevel windows instead, so no NSAlert is ever created.

Supports the subset of the messagebox API this project uses:
showerror(title, message, parent=None), showinfo(title, message, parent=None),
askyesno(title, message, parent=None).
"""

import tkinter as tk

_BG = "#171817"
_FG = "#ededed"
_BUTTON_BG = "#2b2c2d"
_ACCENT_BG = "#416b88"


def _dialog(parent, title, message, buttons):
    owner = parent or tk._default_root
    win = tk.Toplevel(owner)
    win.title(title)
    win.configure(bg=_BG)
    win.resizable(False, False)
    if owner is not None:
        win.transient(owner)

    result = {"value": None}

    def choose(value):
        result["value"] = value
        win.destroy()

    frame = tk.Frame(win, bg=_BG, padx=24, pady=20)
    frame.pack(fill="both", expand=True)

    tk.Label(
        frame,
        text=message,
        bg=_BG,
        fg=_FG,
        justify="left",
        wraplength=360,
        font=("Helvetica", 13),
    ).pack(anchor="w", pady=(0, 18))

    button_row = tk.Frame(frame, bg=_BG)
    button_row.pack(fill="x")

    for i, (text, value) in enumerate(buttons):
        # macOS Tk buttons ignore custom bg/fg fill -- differentiate with a
        # colored highlight ring instead, same as the app's _cnn_button.
        is_primary = i == 0
        btn = tk.Button(
            button_row,
            text=text,
            command=lambda v=value: choose(v),
            fg="#111111",
            activeforeground="#000000",
            relief="raised",
            bd=1,
            padx=16,
            pady=6,
            font=("Helvetica", 12, "bold" if is_primary else "normal"),
            highlightbackground=_ACCENT_BG if is_primary else _BUTTON_BG,
            highlightcolor=_ACCENT_BG if is_primary else _BUTTON_BG,
            highlightthickness=2,
        )
        btn.pack(side="right", padx=(8, 0))
        if is_primary:
            btn.focus_set()

    win.protocol("WM_DELETE_WINDOW", lambda: choose(None))
    win.bind("<Return>", lambda e: choose(buttons[0][1]))
    win.bind("<Escape>", lambda e: choose(None))

    win.update_idletasks()
    if owner is not None:
        x = owner.winfo_rootx() + (owner.winfo_width() - win.winfo_width()) // 2
        y = owner.winfo_rooty() + (owner.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    win.grab_set()
    win.wait_window()
    return result["value"]


def showerror(title, message, parent=None):
    _dialog(parent, title, message, [("OK", True)])


def showinfo(title, message, parent=None):
    _dialog(parent, title, message, [("OK", True)])


def askyesno(title, message, parent=None):
    return bool(_dialog(parent, title, message, [("Yes", True), ("No", False)]))
