# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import types
from typing import List

from detectron2.utils.logger import log_first_n

__all__ = ["DatasetCatalog", "MetadataCatalog"]


class DatasetCatalog(object):
    """
    A catalog that stores information about the data and how to obtain them.

    It contains a mapping from strings
    (which are names that identify a dataset, e.g. "coco_2014_train")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]`.

    The returned dicts should be in Detectron2 Dataset format (See DATASETS.md for details)
    if used with the data loader functionalities in `data/build.py,data/detection_transform.py`.

    The purpose of having this catalog is to make it easy to choose
    different data, by just using the strings in the config.
    """

    _REGISTERED = {}

    @staticmethod
    def register(name, func):
        """
        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
            func (callable): a callable which takes no arguments and returns a list of dicts.
        """
        assert callable(func), "You must register a function with `DatasetCatalog.register`!"
        assert name not in DatasetCatalog._REGISTERED, "Dataset '{}' is already registered!".format(
            name
        )
        DatasetCatalog._REGISTERED[name] = func

    @staticmethod
    def get(name):
        """
        Call the registered function and return its results.

        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".

        Returns:
            list[dict]: dataset annotations.0
        """
        try:
            f = DatasetCatalog._REGISTERED[name]
        except KeyError:
            raise KeyError(
                "Dataset '{}' is not registered! Available data are: {}".format(
                    name, ", ".join(DatasetCatalog._REGISTERED.keys())
                )
            )
        return f()

    @staticmethod
    def list() -> List[str]:
        """
        List all registered data.

        Returns:
            list[str]
        """
        return list(DatasetCatalog._REGISTERED.keys())

    @staticmethod
    def clear():
        """
        Remove all registered dataset.
        """
        DatasetCatalog._REGISTERED.clear()


class Metadata(types.SimpleNamespace):
    """
    A class that supports simple attribute setter/getter.
    It is intended for storing metadata of a dataset and make it accessible globally.

    Examples:

    .. code-block:: python

        # somewhere when you load the data:
        MetadataCatalog.get("mydataset").thing_classes = ["person", "dog"]

        # somewhere when you print statistics or visualize:
        classes = MetadataCatalog.get("mydataset").thing_classes
    """

    # the name of the dataset
    # set default to N/A so that `self.name` in the errors will not trigger getattr again
    name: str = "N/A"

    _RENAMED = {
        "class_names": "thing_classes",
        "dataset_id_to_contiguous_id": "thing_dataset_id_to_contiguous_id",
        "stuff_class_names": "stuff_classes",
    }

    def __getattr__(self, key):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,
            )
            return getattr(self, self._RENAMED[key])

        raise AttributeError(
            "Attribute '{}' does not exist in the metadata of '{}'. Available keys are {}.".format(
                key, self.name, str(self.__dict__.keys())
            )
        )

    def __setattr__(self, key, val):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,
            )
            setattr(self, self._RENAMED[key], val)

        # Ensure that metadata of the same name stays consistent
        try:
            oldval = getattr(self, key)
            assert oldval == val, (
                "Attribute '{}' in the metadata of '{}' cannot be set "
                "to a different value!\n{} != {}".format(key, self.name, oldval, val)
            )
        except AttributeError:
            super().__setattr__(key, val)

    def as_dict(self):
        """
        Returns all the metadata as a dict.
        Note that modifications to the returned dict will not reflect on the Metadata object.
        """
        return copy.copy(self.__dict__)

    def set(self, **kwargs):
        """
        Set multiple metadata with kwargs.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def get(self, key, default=None):
        """
        Access an attribute and return its value if exists.
        Otherwise return default.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class MetadataCatalog:
    """
    MetadataCatalog provides access to "Metadata" of a given dataset.

    The metadata associated with a certain name is a singleton: once created,
    the metadata will stay alive and will be returned by future calls to `get(name)`.

    It's like global variables, so don't abuse it.
    It's meant for storing knowledge that's constant and shared across the execution
    of the program, e.g.: the class names in COCO.
    """

    _NAME_TO_META = {}

    @staticmethod
    def get(name):
        """
        Args:
            name (str): name of a dataset (e.g. coco_2014_train).

        Returns:
            Metadata: The :class:`Metadata` instance associated with this name,
            or create an empty one if none is available.
        """
        assert len(name)
        if name in MetadataCatalog._NAME_TO_META:
            ret = MetadataCatalog._NAME_TO_META[name]
            # TODO this is for the BC breaking change in D15247032.
            # Remove this in the future.
            if hasattr(ret, "dataset_name"):
                logger = logging.getLogger()
                logger.warning(
                    """
The 'dataset_name' key in metadata is no longer used for
sharing metadata among splits after D15247032! Add
metadata to each split (now called dataset) separately!
                    """
                )
                parent_meta = MetadataCatalog.get(ret.dataset_name).as_dict()
                ret.set(**parent_meta)
            return ret
        else:
            m = MetadataCatalog._NAME_TO_META[name] = Metadata(name=name)
            return m

    @staticmethod
    def list():
        """
        List all registered metadata.

        Returns:
            list[str]: keys (names of data) of all registered metadata
        """
        return list(MetadataCatalog._NAME_TO_META.keys())
