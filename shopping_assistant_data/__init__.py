# Package shim to expose top-level modules as shopping_assistant_data.<module>
# This allows tests to import shopping_assistant_data.orchestrator while the modules
# live at the repository root (as single-file modules).
import importlib
import sys
from pathlib import Path

# Modules to expose; add any top-level modules that tests may import as shopping_assistant_data.<name>
MODULES = [
    'orchestrator', 'conversation_state', 'repository', 'llm_nlu', 'nlu_schema', 'schemas', 'repository',
    'orchestrator_features', 'llm_adapter', 'analysis', 'api', 'validators', 'freshness', 'conversation_state',
    # expose new modules implemented at project root
    'product_search', 'text_utils'
]

_pkg_name = __name__
for m in MODULES:
    try:
        mod = importlib.import_module(m)
        # register as submodule shopping_assistant_data.<m>
        fullname = f"{_pkg_name}.{m}"
        sys.modules[fullname] = mod
        # expose as attribute
        globals()[m] = mod
    except Exception:
        # ignore import errors here; tests will raise if module actually needed
        pass
