import multiprocessing.pool
import threading
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


_TK_THREAD = None

_TK_LOCK = threading.Lock()
_TK_EXEC_QUEUE = []
_TK_ACTIVE_FUTURES = set()

_TK_ROOT = None


def _start_tk_thread_if_necessary(tk):
    global _TK_THREAD
    if _TK_THREAD is None:

        def tk_loop():
            global _TK_ROOT, _TK_THREAD
            _TK_ROOT = tk.Tk()
            _TK_ROOT.withdraw()

            def process_queue():
                with _TK_LOCK:
                    for item in _TK_EXEC_QUEUE:
                        _TK_ROOT.after_idle(item)
                    _TK_EXEC_QUEUE.clear()

                    for f in list(_TK_ACTIVE_FUTURES):
                        if f.is_done():
                            _TK_ACTIVE_FUTURES.remove(f)

                    if len(_TK_ACTIVE_FUTURES) == 0:
                        print("INFO: no more jobs, destroying tk root")
                        _TK_ROOT.after_idle(_TK_ROOT.destroy)
                    else:
                        _TK_ROOT.after(100, process_queue)

            _TK_ROOT.after(100, process_queue)
            _TK_ROOT.mainloop()

            _TK_ROOT = None
            _TK_THREAD = None

        print("INFO: starting new tk thread")
        _TK_THREAD = threading.Thread(target=tk_loop)
        _TK_THREAD.start()

def run_on_tk_thread(tk, func, futs=()):
    with _TK_LOCK:
        _TK_EXEC_QUEUE.append(func)
        for f in futs:
            _TK_ACTIVE_FUTURES.add(f)
    _start_tk_thread_if_necessary(tk)

def end_tk_thread():
    with _TK_LOCK:
        for f in _TK_ACTIVE_FUTURES:
            f.set_val(None)


def prompt_for_directory(
        title=None,
        mustexist=True,
        initialdir=None) -> Future:
    return _prompt_for_path(
        title=title,
        mustbefile=False,
        mustexist=mustexist,
        initialdir=initialdir
    )

def prompt_for_filename(
        title=None,
        mustexist=True,
        defaultextension=None,
        filetypes=None,
        initialdir=None,
        initialfile=None,
        multiple=False) -> Future:
    return _prompt_for_path(
        title=title,
        mustbefile=True,
        mustexist=mustexist,
        defaultextension=defaultextension,
        filetypes=filetypes,
        initialdir=initialdir,
        initialfile=initialfile,
        multiple=multiple if mustexist else None)

def _prompt_for_path(
        title=None,
        mustbefile=True,
        mustexist=True,
        defaultextension=None,
        filetypes=None,
        initialdir=None,
        initialfile=None,
        multiple=None) -> Future:
    try:
        import tkinter as tk
        import tkinter.filedialog
    except ImportError as err:
        traceback.print_exc()
        print("ERROR: Couldn't import tkinter, returning None, hope this wasn't important...")
        return Future().set_val(None, err=err)

    fut = Future()

    options = {
        'title': title,
        'mustexist': mustexist,
        'defaultextension': defaultextension,
        'filetypes': filetypes,
        'initialdir': initialdir,
        'initialfile': initialfile,
        'multiple': multiple
    }
    for k, v in dict(options).items():
        if v is None or (mustbefile and k == 'mustexist'):
            del options[k]

    def show_file_prompt():
        try:
            if mustbefile:
                if mustexist:
                    val = tkinter.filedialog.askopenfilename(**options)
                else:
                    val = tkinter.filedialog.asksaveasfilename(**options)
            else:
                val = tkinter.filedialog.askdirectory(**options)
        except Exception as e:
            print("ERROR: file prompt threw error, returning None")
            fut.set_val(None, err=e)
            traceback.print_exc()
        else:
            print(f"INFO: file prompt returned: {val}")
            fut.set_val(val)

    run_on_tk_thread(tk, show_file_prompt, futs=(fut,))
    return fut


def prompt_for_text(window_title, question_text, default_text) -> Future:
    try:
        import tkinter as tk
    except ImportError as err:
        traceback.print_exc()
        print("ERROR: Couldn't import tkinter, returning None, hope this wasn't important...")
        return Future().set_val(None, err=err)

    fut = Future()

    def show_text_prompt():
        window = tk.Toplevel(_TK_ROOT)
        window.title(window_title)

        def on_close():
            print("INFO: text prompt returned: None")
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
            val = text_box.get('1.0', 'end-1c')
            print(f"INFO: text prompt returned: {val}")
            fut.set_val(val)
            window.destroy()

        ok_btn = tk.Button(master=window, text="Ok", command=confirm)
        ok_btn.pack()

    run_on_tk_thread(tk, show_text_prompt, futs=(fut,))

    return fut

def kill_all_worker_threads(include_tk=True):
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None

    if include_tk:
        end_tk_thread()
