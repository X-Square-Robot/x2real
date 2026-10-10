"""Factory interface — lifecycle contract only.

Generator discovery is handled by the global lazy registry.
Instance-level filtering (e.g. _excluded) can be added by
subclasses when parallel isolation is needed.
"""

from abc import ABC
from typing import Any


class BaseFactory(ABC):
    """Pure interface for all pipeline factories.

    Subclasses implement the four lifecycle methods:
        setup -> receive_orders -> fulfill_orders -> cleanup
    """

    def setup(self, cfg: "BaseFactoryCFG") -> None:
        pass

    def receive_orders(self, orders: Any) -> None:
        raise NotImplementedError

    def fulfill_orders(self, **kwargs) -> Any:
        raise NotImplementedError

    def cleanup(self) -> None:
        pass


class BaseFactoryCFG:
    pass
