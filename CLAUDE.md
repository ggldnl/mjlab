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

Comments do not have a dot at the end. Normal comments start with capital letter, 
inline they start with lowercase letter. Avoid at all costs to pad comments to fill
the line with whatever character. Avoid at all costs to box comments.

Examples of this to avoid (python):

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

Apply the same concepts to any other programming language other than python. 
If the user edits some comment or decides to apply any other styling guide,
let it do so, do not change the comments it produces. What I specified only
applies to you.

# Commits

For any big change you produce on a repo, at the end of your message
suggest a commit message that describes the change.
You might need to work on more than one repo at a time: you'll have to do
this for all of them producing a dict {repo: commit message}
Commit messages should be just small sentences, not explain thoroughly
the change.

Example:

```
repo x: "Fixed this problem"
repo y: "Fixed this other problem"
```

Remember: 
- not all the changes you do to a repo are worth having a dedicated commit. 
  Some changes might be extended with the conversation going furhter;
  in that case, omit the commit message suggestion.
- commit message should be plain, concise prose. No section headers, checklists,
  or structured templates. Describe the problem, what the change does, and
  any non-obvious tradeoffs. A good description reads like a short
  paragraph to a colleague, not a form.
- commit messages are rendered on GitHub, so don't hard-wrap them
  at 88 columns. Let each sentence flow on one line.
