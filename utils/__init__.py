# Utils package

# Web annotation (optional, requires Flask)
try:
    from .web_annotation import init_app
    __all__ = ['init_app']
except ImportError:
    __all__ = []

