from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Callable, Iterable, Tuple, Any, Dict, Union
import copy, jsonpickle, json
import numpy as np, pandas as pd, scipy.sparse as sp
from PIL import Image
import pickle
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
import os, re, builtins
from contextlib import contextmanager

Cond = Callable[[DataBundle, "AnimalBlock"], bool]

class AnimalBlock(ABC):
    """
    Base-class for every pipeline block.

    Each concrete subclass must define

        • needs              – immutable prerequisites (tuple[str, ...])
        • provides           – keys it guarantees to add (tuple[str, ...])
        • conditional_needs  – list[ (key, condition(bundle, self)) ]
    """
    _CKPT_FILE = "__checkpoints__.json"

    # ──────────────────────────────────────────────────────────────────
    # constructor
    # ──────────────────────────────────────────────────────────────────
    def __init__(
        self,
        *,
        needs: Sequence[str] = (),
        provides: Sequence[str] = (),
        conditional_needs: Sequence[Tuple[str, Cond]] = (),
        checkpoint_keys: Sequence[str] | None = None,
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        tag: str | None = None,
        verbose: bool = True,
        checkpoint: bool = True,         
        load_checkpoint: bool = True,     
        **attrs: Any,
    ):
        self.needs: Tuple[str, ...] = tuple(needs)
        self.provides: Tuple[str, ...] = tuple(provides)
        self.conditional_needs = list(conditional_needs)
        self.init_settings = init_settings or {}
        self.call_settings = call_settings or {}
        self.tag: str = tag or self.__class__.__name__
        self.verbose = bool(verbose)
        self.checkpoint = bool(checkpoint)
        self.load_checkpoint = bool(load_checkpoint)

        self._ckpt_keys = set(checkpoint_keys) if checkpoint_keys else set(self.provides)



        if hasattr(self.__class__, "CANONICAL_NEEDS"):
            self._canonical_needs = tuple(self.__class__.CANONICAL_NEEDS)
        else:
            self._canonical_needs = tuple(n.split(".", 1)[-1] for n in self.needs)

        for k, v in attrs.items():
            setattr(self, k, v)

        if self.verbose:
            runtime_needs = list(self.needs) + [k for k, _ in self.conditional_needs]
            print(f"[{self.tag}] needs → ({', '.join(runtime_needs) or '∅'})"
                  f"   provides → ({', '.join(self.provides) or '∅'})")

    # ─────────────────────── checkpoint helpers ───────────────────────
    def _ckpt_dir(self, bundle: DataBundle) -> Path:
        return Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag

    def _ckpt_path(self, bundle: DataBundle) -> Path:
        return self._ckpt_dir(bundle) / self._CKPT_FILE

    def _load_ckpt_map(self, bundle: DataBundle) -> Dict[str, str]:
        fp = self._ckpt_path(bundle)
        if fp.is_file():
            try:
                return json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_ckpt_map(self, bundle: DataBundle, ckpt_map: Dict[str, str]) -> None:
        if not self.checkpoint:
            return
        fp = self._ckpt_path(bundle)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(ckpt_map, indent=2), encoding="utf-8")

    def register_checkpoint(self, bundle_key: str, path: Union[str, Path]) -> None:
        """
        Call inside `run()` for every provide you persist to disk.

        Example
        -------
        final_csv = outdir / "results.csv"
        df.to_csv(final_csv, index=False)
        self.register_checkpoint("df", final_csv)
        """
        path = str(path)
        self._pending_ckpt_map[bundle_key] = path

    @staticmethod
    def _merge(default: Dict[str, Any], override: Dict[str, Any] | None):
        return {**default, **(override or {})}
    
    # ───────────────────────────── run cycle ───────────────────────────
    @abstractmethod
    def run(self, bundle: DataBundle) -> None:
        """
        Mutate the bundle in-place; do not return.
        """
        ...

    def __call__(self, bundle: DataBundle) -> DataBundle:
        Path(bundle[SAVE_DIR_BUNDLE_KEY]).mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------------
        # 0) maybe load from checkpoint and bail early
        # --------------------------------------------------------------
        if self.load_checkpoint:
            ckpt = self._load_ckpt_map(bundle)
            have   = self._ckpt_keys.issubset(ckpt.keys())
            exist  = have and all(Path(ckpt[k]).is_file() for k in self._ckpt_keys)
            if exist:
                # hydrate only the keys we cached
                for k in self._ckpt_keys:
                    path = ckpt[k]
                    try:
                        bundle[f"{self.tag}.{k}"] = self.load_path(path)
                    except Exception:
                        # If reading (e.g., empty CSV) fails, just provide the path string.
                        # This still counts as "loaded", so run() is skipped.
                        bundle[f"{self.tag}.{k}"] = path

                # load any 'provides' from the class that is not saved to disk
                self._after_checkpoint_skip(bundle)

                if self.verbose:
                    print(f"[{self.tag}] ✔ loaded from checkpoint")
                # ensure *all* provides are present (lazy ones may already
                # be in the bundle from a previous block run)
                return bundle

        # --------------------------------------------------------------
        # 1) normal validation of needs
        # --------------------------------------------------------------
        runtime_needs = set(self.needs)
        for key, cond in self.conditional_needs:
            if cond(bundle, self):
                runtime_needs.add(key)
        missing = [k for k in runtime_needs if k not in bundle]
        if missing:
            raise KeyError(f"{self.tag}: missing required keys {missing}")

        # --------------------------------------------------------------
        # 2) run block
        # --------------------------------------------------------------
        self._pending_ckpt_map: Dict[str, str] = {}
        with self._io_path_rewriter():
            self.run(bundle)

        # --------------------------------------------------------------
        # 3) verify outputs
        # --------------------------------------------------------------
        view = bundle.namespaced(self.tag)
        missing_out = [k for k in self.provides if k not in view]
        if missing_out:
            raise KeyError(f"{self.tag}: failed to provide {missing_out}")

        # --------------------------------------------------------------
        # 4) persist checkpoints (only those you registered)
        # --------------------------------------------------------------
        if self.checkpoint:
            if self._ckpt_keys.issubset(self._pending_ckpt_map):
                self._save_ckpt_map(bundle, {
                    k: self._pending_ckpt_map[k] for k in self._ckpt_keys
                })
                if self.verbose:
                    print(f"[{self.tag}] ⭳ checkpoint saved")

        return bundle
    
    def _after_checkpoint_skip(self, bundle: DataBundle) -> None:
        # no-op: do nothing by default
        pass

    def save_settings(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        serialized = jsonpickle.encode(self, indent=2)
        p.write_text(serialized, encoding="utf-8")

    def load_settings(self, path: Union[str, Path]) -> None:
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        state = jsonpickle.decode(raw)
        if not hasattr(state, '__dict__'):
            raise ValueError(f"Bad settings format: expected object, got {type(state)}")
        self.__dict__.update(state.__dict__)
        self.settings = state

    def print_settings(self) -> None:
        for key, val in (vars(self).items()):
            print(f"{key}: {val!r}")
        
        print('needs')
        for  i in  self.needs:
            print(f"\t{i!r}") 

        print('provides')
        for  i in  self.provides:
            print(f"\t{i!r}")


    def load_path(self, path: Union[str, Path]) -> Any:
        # In-memory objects pass straight through
        if type(path) in [pd.DataFrame, sp._csr.csr_matrix, np.ndarray, dict, list]:
            return path

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        ext = p.suffix.lower()

        if ext in {".csv", ".tsv"}:
            return pd.read_csv(p)

        if ext == ".json":
            return json.loads(p.read_text(encoding="utf-8"))

        if ext in {".txt", ".html", ".htm"}:
            return p.read_text(encoding="utf-8")

        if ext == ".npy":
            return np.load(p, allow_pickle=True)

        if ext == ".npz":
            try:
                return sp.load_npz(str(p))
            except Exception:
                return np.load(p, allow_pickle=True)

        # ─── Pickle / gpickle ──────────────────────────────────────
        if ext in {".gpickle", ".pkl", ".pickle"}:
            with p.open("rb") as f:
                return pickle.load(f)

        if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
            return Image.open(p)

        # fallback to raw bytes
        return p.read_bytes()


    def save_path(self, data: Any, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower()

        # ─── Pickle / gpickle ──────────────────────────────────────
        if ext in {".gpickle", ".pkl", ".pickle"}:
            with p.open("wb") as f:
                pickle.dump(data, f)
            return

        # ─── Common data types ─────────────────────────────────────
        if isinstance(data, pd.DataFrame):
            data.to_csv(p, index=False)
            return

        if isinstance(data, (dict, list)):
            with p.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return

        if isinstance(data, (str, int, float)):
            with p.open("w", encoding="utf-8") as f:
                f.write(str(data))
            return

        if isinstance(data, np.ndarray):
            if ext in {".csv", ".txt"}:
                np.savetxt(p, data, delimiter=",")
            else:
                np.save(str(p), data)
            return

        if sp.issparse(data):
            sp.save_npz(str(p), data)
            return

        if isinstance(data, Image.Image):
            data.save(p)
            return

        if isinstance(data, (bytes, bytearray)):
            with p.open("wb") as f:
                f.write(data)
            return

        raise ValueError(f"Unsupported data type: {type(data)}")


    def copy(instance: Any,
            needs: Tuple[str, ...]=None,
            provides: Tuple[str, ...] = None) -> Any:
        """
        Return a deep-copy of `instance` with updated `needs` and `provides`.

        Parameters
        ----------
        instance
            Any object that has `needs` and `provides` attributes.
        new_needs
            Tuple of strings to assign to the copied instance's `needs`.
        new_provides
            Tuple of strings to assign to the copied instance's `provides`.

        Returns
        -------
        Any
            A deepcopy of `instance` with its I/O replaced.
        """
        new_obj = copy.deepcopy(instance)
        if needs:
            new_obj.needs = needs
        if provides:
            new_obj.provides = provides
        return new_obj
    


    def _base_tag(self, tag: str) -> str:
        return re.sub(r'^\d+_', '', tag)

    @contextmanager
    def _io_path_rewriter(self):
        """
        While this block runs, rewrite any filesystem path segment matching
        '/<NN>_<baseTag>/' (or '\\<NN>_<baseTag>\\') to the current self.tag,
        for BOTH reads and writes. Also intercept directory creation to prevent
        re-creating old numbered dirs.
        """
        base_tag = self._base_tag(self.tag)
        target_tag = self.tag

        # match start/middle/end occurrences for both / and \
        fwd_middle = re.compile(rf'(?<=/)\d+_{re.escape(base_tag)}(?=/)')
        fwd_head   = re.compile(rf'^\d+_{re.escape(base_tag)}(?=/)')
        fwd_tail   = re.compile(rf'(?<=/)\d+_{re.escape(base_tag)}$')

        bsl_middle = re.compile(rf'(?<=\\)\d+_{re.escape(base_tag)}(?=\\)')
        bsl_head   = re.compile(rf'^\d+_{re.escape(base_tag)}(?=\\)')
        bsl_tail   = re.compile(rf'(?<=\\)\d+_{re.escape(base_tag)}$')

        def _rewrite_seg(path_str: str) -> str:
            s = path_str
            s = fwd_middle.sub(target_tag, s)
            s = fwd_head.sub(target_tag, s)
            s = fwd_tail.sub(target_tag, s)
            s = bsl_middle.sub(target_tag, s)
            s = bsl_head.sub(target_tag, s)
            s = bsl_tail.sub(target_tag, s)
            return s

        # save originals
        orig_open     = builtins.open
        orig_exists   = os.path.exists
        orig_isfile   = os.path.isfile
        orig_isdir    = os.path.isdir
        orig_mkdir    = os.mkdir
        orig_makedirs = os.makedirs
        orig_path_mkdir = Path.mkdir

        def _open_patched(path, mode='r', *args, **kwargs):
            try:
                p = os.fspath(path)
            except TypeError:
                return orig_open(path, mode, *args, **kwargs)
            p2 = _rewrite_seg(p)
            # for reads: prefer rewritten, then fall back
            if 'r' in mode and not any(ch in mode for ch in 'wax+'):
                try:
                    return orig_open(p2, mode, *args, **kwargs)
                except FileNotFoundError:
                    return orig_open(p, mode, *args, **kwargs)
            # for writes or mixed modes: always use rewritten to avoid recreating old dirs
            return orig_open(p2, mode, *args, **kwargs)

        def _wrap_exists(fn):
            def _f(path):
                try:
                    p = os.fspath(path)
                except TypeError:
                    return fn(path)
                return fn(_rewrite_seg(p))
            return _f

        def _mkdir_patched(path, mode=0o777):
            p = _rewrite_seg(os.fspath(path))
            return orig_mkdir(p, mode)

        def _makedirs_patched(path, mode=0o777, exist_ok=False):
            p = _rewrite_seg(os.fspath(path))
            return orig_makedirs(p, mode=mode, exist_ok=exist_ok)

        def _path_mkdir_patched(self_path: Path, mode=0o777, parents=False, exist_ok=False):
            # emulate Path.mkdir but create at the rewritten location
            p = _rewrite_seg(str(self_path))
            if parents:
                return orig_makedirs(p, mode=mode, exist_ok=exist_ok)
            else:
                try:
                    return orig_mkdir(p, mode)
                except FileExistsError:
                    if exist_ok:
                        return
                    raise

        try:
            builtins.open   = _open_patched
            os.path.exists  = _wrap_exists(orig_exists)
            os.path.isfile  = _wrap_exists(orig_isfile)
            os.path.isdir   = _wrap_exists(orig_isdir)
            os.mkdir        = _mkdir_patched
            os.makedirs     = _makedirs_patched
            Path.mkdir      = _path_mkdir_patched
            yield
        finally:
            builtins.open   = orig_open
            os.path.exists  = orig_exists
            os.path.isfile  = orig_isfile
            os.path.isdir   = orig_isdir
            os.mkdir        = orig_mkdir
            os.makedirs     = orig_makedirs
            Path.mkdir      = orig_path_mkdir
