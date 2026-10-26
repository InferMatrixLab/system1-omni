"""Vela 2.0 wire module: envelope admission plus the pinned executors.

The shared wire contract (protocol, prepared-error mapping, response
finishing) lives here for every Vela 2.0 member; each member's pinned
executor runs the checkpoint's own runtime. See README.md in this
directory for the served contract.
"""
