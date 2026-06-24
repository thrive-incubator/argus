"""Bus formatting helpers (LSL channel layout, OSC address mapping).

Pure helpers only — no ``pylsl``/``python-osc`` import required so they are testable
headlessly. The actual transport adapters live in the (out-of-test-scope) I/O shell.
"""
