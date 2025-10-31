"""Language detection utilities."""

import re
from pathlib import Path
from typing import Optional


# Language mapping based on file extensions
LANGUAGE_MAP = {
    '.py': 'Python',
    '.js': 'JavaScript',
    '.jsx': 'JavaScript',
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript',
    '.java': 'Java',
    '.kt': 'Kotlin',
    '.scala': 'Scala',
    '.cpp': 'C++',
    '.cc': 'C++',
    '.cxx': 'C++',
    '.c': 'C',
    '.h': 'C',
    '.cs': 'C#',
    '.php': 'PHP',
    '.rb': 'Ruby',
    '.go': 'Go',
    '.rs': 'Rust',
    '.swift': 'Swift',
    '.m': 'Objective-C',
    '.mm': 'Objective-C++',
    '.r': 'R',
    '.R': 'R',
    '.sh': 'Shell',
    '.bash': 'Bash',
    '.zsh': 'Shell',
    '.ps1': 'PowerShell',
    '.sql': 'SQL',
    '.html': 'HTML',
    '.css': 'CSS',
    '.scss': 'SCSS',
    '.sass': 'Sass',
    '.less': 'Less',
    '.vue': 'Vue',
    '.dart': 'Dart',
    '.lua': 'Lua',
    '.perl': 'Perl',
    '.pl': 'Perl',
    '.pm': 'Perl',
    '.clj': 'Clojure',
    '.hs': 'Haskell',
    '.elm': 'Elm',
    '.ex': 'Elixir',
    '.exs': 'Elixir',
    '.erl': 'Erlang',
    '.hrl': 'Erlang',
    '.vim': 'VimL',
    '.tex': 'TeX',
    '.md': 'Markdown',
    '.json': 'JSON',
    '.xml': 'XML',
    '.yaml': 'YAML',
    '.yml': 'YAML',
    '.toml': 'TOML',
    '.ini': 'INI',
    '.cfg': 'Config',
    '.tf': 'HCL',  # Terraform
    '.dockerfile': 'Dockerfile',
}


def detect_language(filepath: Optional[str] = None, filename: Optional[str] = None) -> str:
    """
    Detect programming language from filepath or filename.
    
    Args:
        filepath: Full file path
        filename: Just the filename
        
    Returns:
        Detected language name or 'unknown'
    """
    # Try filepath first
    path = filepath if filepath else filename
    if not path:
        return 'unknown'
    
    # Extract extension
    path_obj = Path(path)
    ext = path_obj.suffix
    
    # Look up in language map
    if ext in LANGUAGE_MAP:
        return LANGUAGE_MAP[ext]
    
    # Check for dockerfile (case insensitive)
    if path_obj.name.lower() == 'dockerfile':
        return 'Dockerfile'
    
    # Check for makefile
    if path_obj.name.lower() in ['makefile', 'makefile.am', 'makefile.in']:
        return 'Makefile'
    
    # Check for cmake
    if path_obj.name.lower() in ['cmakelists.txt']:
        return 'CMake'
    
    return 'unknown'


def detect_language_from_diff(diff: Optional[str]) -> str:
    """
    Detect language from diff content.
    
    Args:
        diff: Diff string content
        
    Returns:
        Detected language or 'unknown'
    """
    if not diff:
        return 'unknown'
    
    # Look for file paths in diff headers
    lines = diff.split('\n')
    for line in lines[:20]:  # Check first 20 lines
        if line.startswith('+++') or line.startswith('---'):
            # Extract filename
            match = re.search(r'[+-]{3}\s+.*?/([^/\s]+\.\w+)', line)
            if match:
                filename = match.group(1)
                lang = detect_language(filename=filename)
                if lang != 'unknown':
                    return lang
    
    return 'unknown'


