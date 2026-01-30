def safe_metric(fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] {fn.__name__} failed: {e}")
        return default
