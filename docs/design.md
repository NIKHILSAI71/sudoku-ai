# Design Overview

This project separates the core engine, solver backends, and AI components. The engine exposes a `Board` abstraction and helpers. Solvers implement multiple strategies and share a common interface. The AI module provides data loaders and PyTorch models.

