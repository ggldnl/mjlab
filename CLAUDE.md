# Development Workflow

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

Always run `make check` before committing. This runs formatting, linting,
and type checking. Do not commit code that fails type checking.

Before creating a PR, ensure all checks pass with `make test`.

When making user-facing changes, add an entry to `docs/source/changelog.rst`
under the "Upcoming version (not yet released)" section using
Added/Changed/Fixed categories. Reference issues with `:issue:\`123\``
(renders as a link to the GitHub issue).

# Comments style guide

Use plain comments, for example (python):

```
# This is a comment

def func(*args, **kwargs):  # this is an inline comment
    do_something()  # this is another inline comment

```

Comments should not end with a punctuation mark. Normal comments start with capital letter, 
inline comments start with lowercase letter. Avoid at all costs to pad comments to fill
the line with whatever character. Avoid at all costs to box comments.

Examples of comment styles to avoid:

```

# +---------------------------------------------------------------------+
# |                                                                     |
# |                 Absolutely avoid box comments                       |
# |               (except if the user asks for them)                    |
# |                                                                     |
# +---------------------------------------------------------------------+

# --- Avoid this type of padding ---
# %%% Avoid this type of padding, whatever is the character you use %%%
### Avoid this type of padding ###

# --- Avoid using a character to fill the line --------------------------

```

Use the same writing style in any programming language.

Avoid hyphen, en dashes and em dashes in comments as punctuation marks
(you might still use them in formulas).
Avoid bolding text in docstrings and comments when not extremely necessary.
Avoid using "`" when not necessary.
Avoid adding too much math or reference too many variables and bits of code
in comments, as they make it more difficult to read. Keep a concise style.

# Docstrings style guidelines

The style you adopt usually in docstrings is very cumbersome, heavy, difficult 
to read. Nothing is clear immediately as you look through the docstring, you 
have to read a lot to get a slight idea of what the class/component is intended 
to do. When writing a docstring, use a simpler, more concise, direct way. It 
should be immediately clear what the component they refer to does. There should 
be a dedicated section with instructions on what to run and how. 
Do not use prose at all cost, use a vocabulary close to what programmers use,
direct, simple and efficient. 
Use math in the docstrings only in dedicated spots when needed, not in between 
words, as I find that difficult to read. 
Do not abuse inline math, frequent "`", referencing variables and bits of code.
Do not use bold and/or italics.
Do not use complex terms and jargon, go straight to the point. The docstring
should only describe how the code works, it should not be too long.
