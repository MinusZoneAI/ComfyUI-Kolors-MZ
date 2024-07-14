from typing import Any


class A:
    def __init__(self):
        self.a = 1

    def get_a(self):
        return self.a


class B(A):
    def __init__(self):
        super().__init__()
        self.__name__ = A.__name__
        self.b = 2

    def get_b(self):
        return self.b

    def __getattribute__(self, name: str) -> Any:
        print("getattribute")
        return super().__getattribute__(name)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("call")
        return super().__call__(*args, **kwds)

    def __reduce__(self):
        print("reduce")
        return super().__reduce__()

    def __reduce_ex__(self, protocol: int):
        print("reduce_ex")
        return super().__reduce_ex__(protocol)

    def __repr__(self) -> str:
        print("repr")
        return super().__repr__()

    def __str__(self) -> str:
        print("str")
        return super().__str__()

    def __setattr__(self, name: str, value: Any) -> None:
        print("setattr:", name, value)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        print("delattr")
        super().__delattr__(name)

    def __dir__(self):
        print("dir")
        return super().__dir__()

    def __eq__(self, other: Any) -> bool:
        print("eq")
        return super().__eq__(other)

    def __ne__(self, other: Any) -> bool:
        print("ne")
        return super().__ne__(other)

    def __hash__(self) -> int:
        print("hash")
        return super().__hash__()

    def __bool__(self) -> bool:
        print("bool")
        return super().__bool__()


b = B()
print(b.get_b())
print(type(b))
