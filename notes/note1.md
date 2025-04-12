from animelist for each user get their top 10, and from those top 10 get like genres and other stuff to build a profile about them

too complex to map dataset and get image url, when building the web ui, do it by sending requests from frontend to api using MAL_ID


```
@contextlib.contextmanager
    def suppress_stdout():
        new_stdout = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = new_stdout
        try:
            yield new_stdout
        finally:
            sys.stdout = old_stdout

```
The inner `suppress_stdout` context manager temporarily diverts output from sys.stdout to a dummy stream so that any print statements (or similar outputs) produced during the call to `model.encode` (inside the loop) are suppressed. This keeps the console output clean—avoiding clutter that might interfere with the tqdm progress bar or other logging.

The annotation (or decorator) `@contextlib.contextmanager` is used to convert the generator function into a context manager. This allows you to use the function within a `with` statement, which automatically handles the setup and teardown (via `try/finally`) without needing to define a full context manager class with `__enter__` and `__exit__` methods.