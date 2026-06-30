import re

# Patterns that indicate incomplete or debug code
BANNED_MARKERS = ("todo!", "FIXME", "HACK", "XXX", "placeholder", "dummy")

# High-signal patterns: a bare substring match is enough to flag a file. Only
# DISTINCTIVE provider prefixes belong here — a short fragment like "sk_" would
# substring-match ordinary identifiers (task_, disk_) and block legitimate code,
# so Stripe/AWS use their full prefixes.
SECRET_PATTERNS = (
    "PRIVATE KEY",
    "BEGIN RSA",
    "BEGIN EC",
    "BEGIN DSA",
    "sk_live_",  # Stripe secret (live)
    "sk_test_",  # Stripe secret (test)
    "ghp_",  # GitHub PAT
    "gho_",  # GitHub OAuth
    "ghs_",  # GitHub server-to-server
    "ghu_",  # GitHub user-to-server
    "ghr_",  # GitHub refresh
    "github_pat_",  # GitHub fine-grained PAT
    "glpat-",  # GitLab PAT
    "xoxb-",  # Slack bot token
    "xoxp-",  # Slack user token
    "AIza",  # Google API key
    "AKIA",  # AWS access key id
    "ASIA",  # AWS temporary access key id
)

# Patterns too short to match as a bare substring without colliding with ordinary
# tokens. "sk-" alone substring-matches kebab-case identifiers/URLs (disk-size,
# task-list, /task-…), so an OpenAI key is matched by a boundaried regex anchored
# at a word boundary (excludes di"sk-"/ta"sk-"). The run then qualifies EITHER by
# containing a digit (≥6 chars — catches typical high-entropy keys and short
# fakes) OR by being very long (≥40 chars — catches a rare all-letter real key,
# which is 48+ chars, without matching the short word-like "sk-spinner" CSS-class
# identifiers a human writes).
SECRET_REGEXES = (
    re.compile(
        r"\bsk-(?:(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{6,}|[A-Za-z0-9_-]{40,})"
    ),  # OpenAI
)

# Low-signal credential keys. These appear constantly in ordinary source
# (struct fields, function params, config keys), so they are only flagged when
# assigned a concrete quoted literal, not a variable/env reference.
ASSIGNMENT_SECRET_KEYS = (
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "access_key",
    "token",
)

# Extensions to skip during file scanning
SKIP_DIRS = frozenset(
    {
        ".venv",
        "venv",
        ".git",
        "node_modules",
        "__pycache__",
        "target",
        "build",
        "dist",
        ".tox",
        ".mypy_cache",
        ".eggs",
    }
)

CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".sh",
    }
)

# Secrets leak through config/env files just as readily as source — and a
# planted credential there is never "code", so the code-only scan (G5/G9) misses
# it. G6 therefore scans these in addition to CODE_EXTENSIONS. Banned-marker and
# debug-artifact scans deliberately stay code-only (a TODO in a YAML is fine).
SECRET_SCAN_EXTENSIONS = CODE_EXTENSIONS | frozenset(
    {
        ".env",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".properties",
        ".xml",
        ".tfvars",
    }
)

# Dotfiles whose whole name is the extension (``Path(".env").suffix == ""``), so
# they must be matched by name rather than suffix.
SECRET_SCAN_FILENAMES = frozenset({".env", ".envrc", ".netrc", ".pgpass"})

# File types where ``KEY=value`` is a literal config assignment (so an UNQUOTED
# value can be a real secret), as opposed to source code where an unquoted RHS
# is a variable reference. Used to gate the unquoted-secret heuristic so it never
# fires on .py/.rs/etc and blocks legitimate code.
ENV_LITERAL_EXTENSIONS = frozenset({".env", ".ini", ".properties", ".conf", ".cfg"})
