import multiprocessing.pool
import time
import traceback
import typing


class Future:

    def __init__(self, callback=None):
        self._val: typing.Any = None
        self._done: bool = False
        self._err: typing.Optional[Exception] = None

        self._callback = callback

    def __eq__(self, other):
        return self == other

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"{type(self).__name__}({self._val}, done={self._done}, err={self._err})"

    def set_val(self, val: typing.Any, err=None) -> 'Future':
        self._val = val
        self._err = err
        self._done = True
        if self._callback is not None:
            self._callback(val)
        return self

    def is_done(self):
        return self._done

    def is_error(self):
        return self._err is not None

    def get_val(self) -> typing.Any:
        return self._val

    def get_error(self):
        return self._err

    def wait(self, poll_rate_secs=0.25, time_limit_secs=None) -> typing.Any:
        if self._done:
            return self._val
        else:
            start_time_secs = time.time()
            while not self._done:
                cur_time_secs = time.time()
                if time_limit_secs is not None and cur_time_secs - start_time_secs > time_limit_secs:
                    raise ValueError("Future did not complete within time limit ({} sec)".format(time_limit_secs))
                time.sleep(poll_rate_secs)
            return self._val


_THREAD_ID_COUNTER = 0
_POOL: typing.Optional[multiprocessing.pool.ThreadPool] = None


def do_work_on_background_thread(runnable, args=(), name=None, future=None, log=False) -> Future:
    """
    runnable: () -> val or () -> None
    future: an optional Future to use. If None, a new one will be made.
    """
    if future is None:
        future = Future()

    global _POOL
    if _POOL is None:
        _POOL = multiprocessing.pool.ThreadPool()

    global _THREAD_ID_COUNTER
    name = f"{(name or 'UnnamedThread')} (id={_THREAD_ID_COUNTER})"
    _THREAD_ID_COUNTER += 1

    def _do_work():
        try:
            result = runnable(*args)
        except Exception as err:
            print(f"ERROR: thread {name} threw error")
            traceback.print_exc()
            future.set_val(None, err)
        else:
            if log:
                print(f"INFO: thread {name} finished with result: {result}")
            future.set_val(result)

    if log:
        print(f"INFO: starting thread: {name}")

    _POOL.apply_async(_do_work)

    return future


def kill_all_worker_threads():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


_TK_ROOT = None

def _get_or_create_tk_root_window(tk):
    global _TK_ROOT
    _TK_ROOT = tk.Tk()
    _TK_ROOT.withdraw()

    return _TK_ROOT

def destroy_popup_windows():
    global _TK_ROOT
    if _TK_ROOT is not None:
        _TK_ROOT.destroy()
        _TK_ROOT = None

def update_popup_windows():
    global _TK_ROOT
    if _TK_ROOT is not None:
        _TK_ROOT.update()
        _TK_ROOT.update_idletasks()


def prompt_for_text(window_title, question_text, default_text, do_async=True) -> Future:
    try:
        import tkinter as tk
    except ImportError as err:
        traceback.print_exc()
        print("ERROR: Couldn't import tkinter, returning None, hope this wasn't important...")
        return Future().set_val(None, err=err)

    root = _get_or_create_tk_root_window(tk)

    fut = Future()

    def build_window():
        window = tk.Toplevel(root)
        window.title(window_title)

        def on_close():
            print("Popup window closed")
            window.destroy()
            fut.set_val(None)

        window.protocol("WM_DELETE_WINDOW", on_close)

        greeting = tk.Label(master=window, text=question_text, anchor="w")
        greeting.pack(fill="both")

        text_box = tk.Text(master=window)
        text_box.insert("1.0", default_text)
        text_box.pack()
        text_box.focus_set()

        def confirm():
            fut.set_val(text_box.get('1.0', 'end-1c'))
            window.destroy()

        ok_btn = tk.Button(master=window, text="Ok", command=confirm)
        ok_btn.pack()

    root.after_idle(build_window)

    if do_async:
        return do_work_on_background_thread(lambda: fut.wait(), name="TextPromptPopup", log=True)
    else:
        return fut.wait()
