#   Class to inherit for to_dict/from_dict Serializable classes
from __future__ import annotations
from typing import TYPE_CHECKING

import inspect
import os
import io
import importlib
import shutil
import dill as pickle
import json
from enum import Enum

import narwhals as nw
from .utilities.inputs import create_folders_if_needed, is_narwhals_compatible

from . import logger

if TYPE_CHECKING:
    import polars as pl


class Serializable:
    _save_exclude_items = []
    _save_suffix = "serial"
    _registry = {}

    try:
        import numpy as np

        _BASE_TYPES = (int, float, str, bool, np.float64)
    except ImportError:
        _BASE_TYPES = (int, float, str, bool)

    def __init__(self):
        self.__fully_serializable__ = True

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses when they're defined"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_save_suffix") and cls._save_suffix:
            Serializable._registry[cls._save_suffix] = cls
            logger.debug(f"Registered {cls.__name__} with suffix '{cls._save_suffix}'")

    @classmethod
    def _init_from_dict(
        cls, data: dict, init_kwargs: dict | None = None, **additional_kwargs
    ):
        init_keys = list(inspect.signature(cls.__init__).parameters.keys())
        init_keys.remove("self")
        if init_kwargs is None:
            init_kwargs = {}
        remaining_kwargs = {}
        for key, value in data.items():
            if key in init_keys:
                init_kwargs[key] = value
            else:
                remaining_kwargs[key] = value

        obj = cls(**init_kwargs, **additional_kwargs)

        for keyi, valuei in remaining_kwargs.items():
            if (keyi not in obj._reserved_keys()) and (
                keyi not in obj._save_exclude_items
            ):
                setattr(obj, keyi, valuei)

        return obj

    def __hash__(self):
        d_save = self.to_dict()
        if self.__fully_serializable__:
            save_caller = json
        else:
            save_caller = pickle

        return hash(save_caller.dumps(d_save))

    def save(self, path: str, quietly: bool = True) -> None:
        """
        Save a serializable object to

        Parameters
        ----------
        path : str, optional
            Path of object.  Can exclude the suffix.
        quietly: bool, optional
            No message to console/log
        """

        self.__serializable_n_dfs = 0

        d_save = self.to_dict()

        if type(path) is not str:
            path = str(path)
        if not path.endswith(f".{self._save_suffix}"):
            folder_path = f"{path}.{self._save_suffix}"
        else:
            folder_path = path

        #   logger.info(folder_path)

        if os.path.isdir(folder_path):
            logger.info("Removing existing directory " + folder_path)
            shutil.rmtree(folder_path)

        #   Make the path to save everything
        create_folders_if_needed([folder_path])
        #   os.makedirs(folder_path)

        #   Save the data
        dfs = d_save["__serialized_dfs__"]
        if len(dfs):
            self._save_dfs(folder_path=folder_path, dfs=dfs, quietly=quietly)

        if self.__fully_serializable__:
            suffix = "json"
            save_caller = json
            w = "w"
        else:
            suffix = "pkl"
            save_caller = pickle
            w = "wb"

        path_object = f"{folder_path}/{self._save_suffix}.{suffix}"
        with open(path_object, w) as f:
            save_caller.dump(d_save, f)

    def _save_dfs(self, folder_path: str, dfs: dict, quietly: bool = True) -> None:
        for valuei in dfs.values():
            dfi = valuei["df"]
            pathi = valuei["path"]

            path_save = os.path.normpath(f"{folder_path}/{pathi}")
            create_folders_if_needed([os.path.dirname(path_save)])

            dfi_nw = nw.from_native(dfi)
            d_metadata = dict(engine=nw.get_native_namespace(dfi_nw).__package__)
            SerializableDictionary(d_metadata).save(path_save)
            if isinstance(dfi_nw, nw.LazyFrame):
                dfi_nw.sink_parquet(path_save)
            else:
                dfi_nw.write_parquet(path_save)

            del valuei["df"]

    @classmethod
    def delete(cls, path: str):
        cls.load(path=path, delete=True, delete_only=True)

    @classmethod
    def load(
        cls,
        path: str = "",
        delete: bool = False,
        delete_only: bool = False,
        init_kwargs: dict | None = None,
        **df_kwargs,
    ) -> Serializable | None:
        """
        Load a serializable object from disk

        Parameters
        ----------
        path : str, optional
            Path of object.  Can exclude the suffix.
        delete : bool, optional
            Delete after load? The default is False.
        delete_only : bool, optional
            Don't load, just delete. The default is False.
        init_kwargs : dict, optional
            Arguments to pass to init function
        df_kwargs : dict, optional
            Arguments to pass to narwhals scan_parquet for dataframe loading
        Returns
        -------
        Serializable
        """

        if type(path) is not str:
            path = str(path)

        # If called on Serializable base class, auto-detect the subclass
        if cls == Serializable:
            # Find which suffix matches
            detected_class = None
            for suffix, registered_cls in Serializable._registry.items():
                if path.endswith(f".{suffix}") or os.path.isdir(f"{path}.{suffix}"):
                    detected_class = registered_cls
                    break

            if detected_class is None:
                message = f"Could not detect serializable type for path: {path}"
                logger.error(message)
                raise Exception(message)

            # Delegate to the correct class
            return detected_class.load(
                path=path, delete=delete, delete_only=delete_only, **df_kwargs
            )

        if not path.endswith(f".{cls._save_suffix}"):
            folder_path = f"{path}.{cls._save_suffix}"
        else:
            folder_path = path

        if not delete_only:
            path_dict = os.path.normpath(f"{folder_path}/{cls._save_suffix}.json")
            path_found = os.path.isfile(path_dict)

            if path_found:
                load_caller = json
            else:
                path_dict = os.path.normpath(f"{folder_path}/{cls._save_suffix}.pkl")
                load_caller = pickle
                path_found = os.path.isfile(path_dict)

            if not path_found:
                message = f"{folder_path} isn't a valid Serializable object"
                logger.error(message)
                raise Exception(message)

            with open(path_dict, "rb") as f:
                d_loaded = load_caller.load(f)

            if len(d_loaded["__serialized_dfs__"]):
                cls._load_dfs(
                    folder_path=folder_path,
                    dfs=d_loaded["__serialized_dfs__"],
                    delete=delete,
                    **df_kwargs,
                )

        if delete or delete_only:
            if os.path.isdir(folder_path):
                logger.info("Removing existing directory " + folder_path)
                shutil.rmtree(folder_path)

        if delete_only:
            return None

        obj = cls.from_dict(d_loaded, init_kwargs=init_kwargs)
        return obj

    @classmethod
    def load_any(cls, path: str = "", delete: bool = False, delete_only: bool = False):
        """
        Pass the root path of a serializable object
            and this will figure out what it is (from the suffix)
            and call that object's loader function.

            For cases where multiple objects work, and I don't want to
            have to do if/else logic by suffix

        Parameters
        ----------
        path : str, optional
            Path of object.  Can exclude the suffix.
        delete : bool, optional
            Delete after load? The default is False.
        delete_only : bool, optional
            Don't load, just delete. The default is False.

        Returns
        -------
        Serializable object of any type

        """

        items = []

        for classi in items:
            suffixi = classi._save_suffix
            if path.endswith(suffixi) or os.path.isdir(f"{path}.{suffixi}"):
                return classi.load(path=path, delete=delete, delete_only=delete_only)

    @classmethod
    def _load_dfs(
        cls, folder_path: str, dfs: dict, delete: bool = False, **df_kwargs
    ) -> None:
        #   Avoid circular import
        from .utilities.dataframe import NarwhalsType

        if df_kwargs is None:
            df_kwargs = {}
        for keyi, valuei in dfs.items():
            pathi = valuei["path"]

            path_load = os.path.normpath(f"{folder_path}/{pathi}")
            d_metadata = SerializableDictionary.load(path_load)
            df_kwargsi = df_kwargs.copy()
            if "backend" not in df_kwargsi:
                df_kwargsi["backend"] = d_metadata["engine"]
                backend = d_metadata["engine"]
            else:
                backend = df_kwargsi["backend"]
            dfi = nw.scan_parquet(path_load, **df_kwargsi)
            if delete:
                dfi = dfi.lazy().collect().lazy_backend(NarwhalsType(backend=backend))

            dfs[keyi]["df"] = dfi.to_native()

    def to_dict(self, dfs: dict | None = None, key_path: str = "") -> dict[str, object]:
        candidate_items = vars(self)

        self.__fully_serializable__ = True
        items = {}
        if dfs is None:
            add_dfs = True
            dfs = {}
        else:
            add_dfs = False

        for keyi, valuei in candidate_items.items():
            self._to_dict_item(
                items=items, dfs=dfs, key_path=key_path, key=keyi, value=valuei
            )

        items["__fully_serializable__"] = self.__fully_serializable__
        items["__class__"] = self.__class__.__qualname__
        if add_dfs:
            items["__serialized_dfs__"] = dfs
        else:
            items["__serialized_dfs__"] = {}

        namespace = vars(inspect.getmodule(self))["__name__"]
        items["__namespace__"] = namespace

        #   logger.info(items)
        return items

    @classmethod
    def from_dict(
        cls,
        data: dict,
        init_kwargs: dict | None = None,
    ) -> object:
        data = cls._from_dict_unpack_item(data, dfs=None, init_kwargs=init_kwargs)

        return data

    @classmethod
    def _from_dict_unpack_item(
        cls,
        item: object,
        dfs: dict | None,
        init_kwargs: dict | None = None,
    ) -> object:
        if dfs is None:
            if "__serialized_dfs__" in item.keys():
                dfs = item["__serialized_dfs__"]
                del item["__serialized_dfs__"]
            else:
                dfs = {}

        typei = type(item)

        if typei is dict:
            if "__enumeration__" in item.keys():
                item = cls._from_dict_unpack_enum(item, dfs)
            elif "__pickled__" in item.keys():
                item = cls._from_dict_unpack_pickled(item, dfs)
            # elif "__polars_expression__" in item.keys():
            #     item = cls._from_dict_unpack_polars_expression(item,
            #                                                    dfs)
            elif cls._is_serializable(item):
                #   Unpack as a separate item
                item = cls._from_dict_unpack_dict(item, dfs, init_kwargs=init_kwargs)

                #   Then turn it into a class
                _namespace = importlib.import_module(item["__namespace__"])
                _class_name = item["__class__"]
                _parent = _namespace
                for classi in _class_name.split("."):
                    # #   Fix for any refactor class name changes
                    # d_refactor = {"MultipleImputationStats":"MultipleImputation"}

                    # classi = d_refactor.get(classi,
                    #                         classi)

                    _class = getattr(_parent, classi)
                    _parent = _class

                #   _class = getattr(_namespace, item['__class__'])

                init_sig = inspect.signature(_class._init_from_dict)
                if "init_kwargs" in init_sig.parameters.keys():
                    item = _class._init_from_dict(item, init_kwargs=init_kwargs)
                else:
                    if init_kwargs is None:
                        init_kwargs = {}
                    item = _class._init_from_dict(item, **init_kwargs)

            else:
                item = cls._from_dict_unpack_dict(item, dfs)
        elif typei is list:
            item = cls._from_dict_unpack_list(item, dfs)
        elif typei is str and item.startswith("__serialized_df_"):
            item = cls._from_dict_assign_df(item, dfs)
        elif typei in [int, float, str, bool]:
            #   Do nothing - don't unpack
            pass

        return item

    @classmethod
    def _from_dict_unpack_dict(
        cls, item: dict, dfs: dict, init_kwargs: dict | None = None
    ) -> object:
        for key, value in item.items():
            if key not in cls._reserved_keys():
                item[key] = cls._from_dict_unpack_item(value, dfs, init_kwargs)

        return item

    @classmethod
    def _from_dict_unpack_list(cls, item: list, dfs: dict):
        for i in range(0, len(item)):
            item[i] = cls._from_dict_unpack_item(item[i], dfs)

        return item

    @classmethod
    def _from_dict_unpack_enum(cls, item: dict, dfs: dict):
        #   Then turn it into a class
        _namespace = importlib.import_module(item["__enum_namespace__"])
        _class_name = item["__enum_class__"]
        _parent = _namespace
        for classi in _class_name.split("."):
            _class = getattr(_parent, classi)
            _parent = _class
        value = item["__enumeration__"]

        return _class(value)

    @classmethod
    def _from_dict_unpack_polars_expression(cls, item: dict, dfs: dict):
        try:
            import polars as pl
        except ImportError:
            raise ImportError("Polars is required for unpacking a polars Expression. ")

        return pl.Expr.deserialize(io.StringIO(item["__polars_expression__"]))

    @classmethod
    def _from_dict_unpack_pickled(cls, item: dict, dfs: dict):
        pickled_data = bytes(item["__pickled__"])
        buffer = io.BytesIO(pickled_data)
        try:
            return pickle.load(buffer)
        except:
            logger.info("Pickled item failed to load")

    @classmethod
    def _from_dict_assign_df(cls, item: str, dfs: dict):
        return dfs[item]["df"]

    def _to_dict_item(
        self,
        items: dict[str, object],
        dfs: dict[str, dict],
        key_path: str,
        key: str,
        value: object,
    ) -> None:
        # logger.info(key)
        # logger.info(value)
        # logger.info(items)
        # logger.info(key_path)

        if (key not in self._reserved_keys()) and (key not in self._save_exclude_items):
            typei = type(value)

            if typei is list:
                d_item = []

                for i in range(0, len(value)):
                    self._to_dict_item(
                        items=d_item,
                        dfs=dfs,
                        key_path=f"{key_path}/{key}/{i}",
                        key=i,
                        value=value[i],
                    )

                self._add_to_items(key=key, value=d_item, items=items)
            elif typei is dict:
                d_item = {}

                for ki, vi in value.items():
                    self._to_dict_item(
                        items=d_item,
                        dfs=dfs,
                        key_path=f"{key_path}/{key}",
                        key=ki,
                        value=vi,
                    )

                self._add_to_items(key=key, value=d_item, items=items)
            elif is_narwhals_compatible(value):
                n_dfs = len(dfs.keys())

                df_name = f"__serialized_df_{n_dfs}"
                dfs[df_name] = {
                    "df": value,
                    "path": f"{key_path}/{key}/{df_name}.parquet",
                }

                self._add_to_items(key=key, value=df_name, items=items)
            elif typei in self._BASE_TYPES:
                self._add_to_items(key=key, value=value, items=items)
            # elif isinstance(value,pl.Expr):
            #     self._add_to_items(key=key,
            #                        value=self._add_to_items_polars_expression(value),
            #                        items=items)
            elif isinstance(value, Serializable):
                value_add = value.to_dict(dfs=dfs, key_path=key_path)
                self.__fully_serializable__ = (
                    self.__fully_serializable__ and value.__fully_serializable__
                )
                self._add_to_items(key=key, value=value_add, items=items)
            elif isinstance(value, Enum):
                value_add = self._add_to_items_enumeration(value)
                self._add_to_items(key=key, value=value_add, items=items)
            elif value is None:
                self._add_to_items(key=key, value=value, items=items)
            else:
                # logger.info(f"{key} not serializable")
                # logger.info(value)
                #   self.__fully_serializable__ = False
                self._add_to_items(
                    key=key, value=self._add_to_items_pickled(value), items=items
                )

    def _add_to_items_enumeration(self, value: Enum) -> dict:
        namespace = value.__module__
        # vars(inspect.getmodule(self))["__name__"]

        # logger.info({
        #     "__enumeration__": value.value,
        #     "__enum_class__": value.__class__.__qualname__,
        #     "__enum_namespace__": namespace,
        # })
        # logger.info("")
        return {
            "__enumeration__": value.value,
            "__enum_class__": value.__class__.__qualname__,
            "__enum_namespace__": namespace,
        }

    def _add_to_items_polars_expression(self, value: pl.Expr) -> dict:
        return {"__polars_expression__": value.meta.serialize()}

    def _add_to_items_pickled(self, value: pl.Expr) -> dict:
        buffer = io.BytesIO()
        pickle.dump(value, buffer)
        value_save = list(buffer.getvalue())
        return {"__pickled__": value_save}

    @staticmethod
    def _add_to_items(key: str, value: object, items: dict | list):
        if type(items) is dict:
            items[key] = value
        elif type(items) is list:
            items.append(value)

    @staticmethod
    def _reserved_keys() -> list[str]:
        return [
            "__class__",
            "__namespace__",
            "__fully_serializable__",
            "__dfs__",
            "__dfpath__",
        ]

    @staticmethod
    def _is_serializable(data: dict):
        return (
            ("__fully_serializable__" in data.keys())
            and ("__class__" in data.keys())
            and ("__namespace__" in data.keys())
        )


class SerializableDictionary(Serializable):
    _save_suffix = "dict"

    def __init__(self, _d: dict):
        self._d = _d

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value

    def __delitem__(self, key):
        del self._d[key]

    def __iter__(self):
        return iter(self._d)

    def __repr__(self):
        return repr(self._d)

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()

    #####################################################
    #   Serializable - BEGIN
    #####################################################
    @classmethod
    def load(
        cls, path: str = "", delete: bool = False, delete_only: bool = False
    ) -> Serializable | None:
        args = locals().copy()
        del args["cls"]
        del args["__class__"]
        d = super().load(**args)

        if not delete_only:
            return d._d

    #####################################################
    #   Serializable - END
    #####################################################


class SerializableList(Serializable):
    _save_suffix = "list"

    def __init__(self, _l: list):
        self._l = _l

    def __len__(self):
        return len(self._l)

    def __getitem__(self, index):
        return self._l[index]

    def __setitem__(self, index, value):
        self._l[index] = value

    def __delitem__(self, index):
        del self._l[index]

    def __iter__(self):
        return iter(self._l)

    def __repr__(self):
        return repr(self._l)

    def append(self, item):
        self._l.append(item)

    def extend(self, iterable):
        self._l.extend(iterable)

    def insert(self, index, item):
        self._l.insert(index, item)

    def remove(self, item):
        self._l.remove(item)

    def pop(self, index=-1):
        return self._l.pop(index)

    def clear(self):
        self._l.clear()

    def index(self, item):
        return self._l.index(item)

    def count(self, item):
        return self._l.count(item)

    def sort(self, key=None, reverse=False):
        self._l.sort(key=key, reverse=reverse)

    def reverse(self):
        self._l.reverse()

    #####################################################
    #   Serializable - BEGIN
    #####################################################
    @classmethod
    def load(
        cls, path: str = "", delete: bool = False, delete_only: bool = False
    ) -> Serializable | None:
        args = locals().copy()
        del args["cls"]
        del args["__class__"]
        l = super().load(**args)

        if not delete_only:
            return l._l

    #####################################################
    #   Serializable - END
    #####################################################
