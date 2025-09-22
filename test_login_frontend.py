# test_login_frontend.py
"""
Single-file test frontend for PR/issue automation.
- Login screen (dummy validation)
- Post-login dashboard (simulate issue commit / PR action)
- Fully local, no external dependencies
"""

import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import math

# ---------- Dummy backend user data ----------
# In real usage, this could call your local automation backend
USER_DB = {
    "alice": "password123",
    "bob": "secret",
    "testuser": "testpass"
}

# ---------- Backend simulation ----------
def validate_login(username, password):
    """Simulate backend login check"""
    correct = USER_DB.get(username)
    return correct is not None and correct == password

def commit_issue_simulation(username):
    """Simulate committing an issue or marking task done"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {username} simulated an issue commit.")
    messagebox.showinfo("Commit Success", f"Issue committed by {username} at {timestamp}")

# ---------- GUI ----------
class LoginApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Test Login - GitHub Automation Frontend")
        self.root.geometry("400x250")
        self.root.resizable(False, False)
        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.create_login_screen()

    def create_login_screen(self):
        tk.Label(self.root, text="Welcome! Please login", font=("Arial", 16)).pack(pady=10)

        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Label(frame, text="Username:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(frame, textvariable=self.username_var).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(frame, text="Password:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(frame, textvariable=self.password_var, show="*").grid(row=1, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Login", width=15, command=self.handle_login).pack(pady=10)

        self.status_label = tk.Label(self.root, text="", fg="red")
        self.status_label.pack()

    def handle_login(self):
        username = self.username_var.get().strip()
        password = self.password_var.get().strip()
        if not username or not password:
            self.status_label.config(text="Please enter username and password")
            return
        if validate_login(username, password):
            print(f"[INFO] {username} logged in successfully")
            self.open_dashboard(username)
        else:
            self.status_label.config(text="Invalid credentials")
            print(f"[WARN] Failed login attempt: {username}")

    def open_dashboard(self, username):
        # clear login widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.title(f"Dashboard - {username}")

        tk.Label(self.root, text=f"Hello, {username}!", font=("Arial", 16)).pack(pady=10)
        tk.Label(self.root, text="This is your test dashboard for issue commitment").pack(pady=5)

        tk.Button(self.root, text="Commit Issue", width=20, height=2,
                  command=lambda: commit_issue_simulation(username)).pack(pady=15)

        tk.Button(self.root, text="Logout", width=10, command=self.logout).pack(pady=5)

    def logout(self):
        print("[INFO] User logged out")
        self.username_var.set("")
        self.password_var.set("")
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_login_screen()

# ---------- Main ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = LoginApp(root)
    root.mainloop()
