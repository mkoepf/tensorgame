# levels.py
import numpy as np

levels = [
    # 1. Flatten
    {
        "input": lambda: np.array([[1, 2], [3, 4]]),
        "target": lambda: np.array([1, 2, 3, 4]),
        "target_shape": (4,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([1, 2, 3, 4])),
        "hint": "Use .reshape(-1) or .flatten()",
    },
    # 2. Reshape
    {
        "input": lambda: np.arange(6),
        "target": lambda: np.array([[0, 1, 2], [3, 4, 5]]),
        "target_shape": (2, 3),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([[0, 1, 2], [3, 4, 5]])),
        "hint": "Use .reshape(2, 3)",
    },
    # 3. Transpose
    {
        "input": lambda: np.array([[1, 2, 3], [4, 5, 6]]),
        "target": lambda: np.array([[1, 4], [2, 5], [3, 6]]),
        "target_shape": (3, 2),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([[1, 4], [2, 5], [3, 6]])),
        "hint": "Use .T or .transpose()",
    },
    # 4. Add scalar
    {
        "input": lambda: np.zeros((2, 2)),
        "target": lambda: np.ones((2, 2)),
        "target_shape": (2, 2),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.ones((2, 2))),
        "hint": "Add 1 to x",
    },
    # 5. Multiply
    {
        "input": lambda: np.ones((3,)),
        "target": lambda: np.array([2, 2, 2]),
        "target_shape": (3,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([2, 2, 2])),
        "hint": "Multiply by 2",
    },
    # 6. Broadcast add
    {
        "input": lambda: np.array([[1], [2], [3]]),
        "target": lambda: np.array([[2, 3], [3, 4], [4, 5]]),
        "target_shape": (3, 2),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([[2, 3], [3, 4], [4, 5]])),
        "hint": "Add [1, 2] to x",
    },
    # 7. Slice
    {
        "input": lambda: np.arange(10),
        "target": lambda: np.array([2, 3, 4]),
        "target_shape": (3,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([2, 3, 4])),
        "hint": "Use slicing: x[2:5]",
    },
    # 8. Reverse
    {
        "input": lambda: np.array([1, 2, 3, 4]),
        "target": lambda: np.array([4, 3, 2, 1]),
        "target_shape": (4,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([4, 3, 2, 1])),
        "hint": "Reverse with slicing: x[::-1]",
    },
    # 9. Stack
    {
        "input": lambda: np.array([1, 2]),
        "target": lambda: np.array([[1, 2], [1, 2]]),
        "target_shape": (2, 2),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([[1, 2], [1, 2]])),
        "hint": "Use np.stack or np.tile",
    },
    # 10. Repeat
    {
        "input": lambda: np.array([1, 2, 3]),
        "target": lambda: np.array([1, 1, 2, 2, 3, 3]),
        "target_shape": (6,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([1, 1, 2, 2, 3, 3])),
        "hint": "Use np.repeat(x, 2)",
    },
    # 11. Concatenate
    {
        "input": lambda: np.array([1, 2]),
        "target": lambda: np.array([1, 2, 1, 2]),
        "target_shape": (4,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([1, 2, 1, 2])),
        "hint": "Use np.concatenate([x, x])",
    },
    # 12. Diagonal
    {
        "input": lambda: np.array([1, 2, 3]),
        "target": lambda: np.diag([1, 2, 3]),
        "target_shape": (3, 3),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.diag([1, 2, 3])),
        "hint": "Use np.diag(x)",
    },
    # 13. Identity
    {
        "input": lambda: np.zeros((3,)),
        "target": lambda: np.eye(3),
        "target_shape": (3, 3),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.eye(3)),
        "hint": "Use np.eye(3)",
    },
    # 14. Argmax
    {
        "input": lambda: np.array([1, 3, 2]),
        "target": lambda: np.array(1),
        "target_shape": (),
        "test": lambda r: np.isscalar(r) and r == 1,
        "hint": "Use x.argmax()",
    },
    # 15. Sum
    {
        "input": lambda: np.array([[1, 2], [3, 4]]),
        "target": lambda: np.array(10),
        "target_shape": (),
        "test": lambda r: np.isscalar(r) and r == 10,
        "hint": "Use x.sum()",
    },
    # 16. Mean
    {
        "input": lambda: np.array([1, 2, 3, 4]),
        "target": lambda: np.array(2.5),
        "target_shape": (),
        "test": lambda r: np.isscalar(r) and np.isclose(r, 2.5),
        "hint": "Use x.mean()",
    },
    # 17. Max along axis
    {
        "input": lambda: np.array([[1, 5], [3, 2]]),
        "target": lambda: np.array([3, 5]),
        "target_shape": (2,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([3, 5])),
        "hint": "Use x.max(axis=0)",
    },
    # 18. Min along axis
    {
        "input": lambda: np.array([[1, 5], [3, 2]]),
        "target": lambda: np.array([1, 2]),
        "target_shape": (2,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([1, 2])),
        "hint": "Use x.min(axis=0)",
    },
    # 19. Argmin along axis
    {
        "input": lambda: np.array([[1, 5], [3, 2]]),
        "target": lambda: np.array([0, 1]),
        "target_shape": (2,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([0, 1])),
        "hint": "Use x.argmin(axis=0)",
    },
    # 20. Boolean mask
    {
        "input": lambda: np.array([1, 2, 3, 4]),
        "target": lambda: np.array([2, 4]),
        "target_shape": (2,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([2, 4])),
        "hint": "Use x[x % 2 == 0]",
    },
    # 21. Where
    {
        "input": lambda: np.array([1, 2, 3]),
        "target": lambda: np.array([10, 2, 10]),
        "target_shape": (3,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([10, 2, 10])),
        "hint": "Use np.where(x == 2, x, 10)",
    },
    # 22. Clip
    {
        "input": lambda: np.array([1, 5, 10]),
        "target": lambda: np.array([2, 5, 8]),
        "target_shape": (3,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([2, 5, 8])),
        "hint": "Use np.clip(x, 2, 8)",
    },
    # 23. Pad
    {
        "input": lambda: np.array([1, 2, 3]),
        "target": lambda: np.array([0, 1, 2, 3, 0]),
        "target_shape": (5,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([0, 1, 2, 3, 0])),
        "hint": "Use np.pad(x, (1, 1))",
    },
    # 24. Tile
    {
        "input": lambda: np.array([1, 2]),
        "target": lambda: np.array([1, 2, 1, 2, 1, 2]),
        "target_shape": (6,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([1, 2, 1, 2, 1, 2])),
        "hint": "Use np.tile(x, 3)",
    },
    # 25. Expand dims
    {
        "input": lambda: np.array([1, 2, 3]),
        "target": lambda: np.array([[1, 2, 3]]),
        "target_shape": (1, 3),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([[1, 2, 3]])),
        "hint": "Use np.expand_dims(x, 0) or x[None, :]",
    },
    # 26. Squeeze
    {
        "input": lambda: np.ones((1, 3, 1)),
        "target": lambda: np.ones((3,)),
        "target_shape": (3,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.ones((3,))),
        "hint": "Use np.squeeze(x)",
    },
    # 27. Swap axes
    {
        "input": lambda: np.arange(8).reshape(2, 2, 2),
        "target": lambda: np.arange(8).reshape(2, 2, 2).swapaxes(0, 2),
        "target_shape": (2, 2, 2),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.arange(8).reshape(2, 2, 2).swapaxes(0, 2)),
        "hint": "Use x.swapaxes(0, 2)",
    },
    # 28. Roll
    {
        "input": lambda: np.array([1, 2, 3, 4]),
        "target": lambda: np.array([4, 1, 2, 3]),
        "target_shape": (4,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([4, 1, 2, 3])),
        "hint": "Use np.roll(x, 1)",
    },
    # 29. Cumsum
    {
        "input": lambda: np.array([1, 2, 3]),
        "target": lambda: np.array([1, 3, 6]),
        "target_shape": (3,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([1, 3, 6])),
        "hint": "Use x.cumsum()",
    },
    # 30. Outer product
    {
        "input": lambda: np.array([1, 2]),
        "target": lambda: np.array([[1, 2], [2, 4]]),
        "target_shape": (2, 2),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([[1, 2], [2, 4]])),
        "hint": "Use np.outer(x, x)",
    },
    # 31. Dot product
    {
        "input": lambda: np.array([1, 2, 3]),
        "target": lambda: np.array(14),
        "target_shape": (),
        "test": lambda r: np.isscalar(r) and r == 14,
        "hint": "Use x.dot(x)",
    },
    # 32. Matrix multiplication
    {
        "input": lambda: np.array([[1, 2], [3, 4]]),
        "target": lambda: np.array([[7, 10], [15, 22]]),
        "target_shape": (2, 2),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([[7, 10], [15, 22]])),
        "hint": "Use x @ x",
    },
    # 33. Broadcast multiply
    {
        "input": lambda: np.arange(3),
        "target": lambda: np.array([0, 2, 4]),
        "target_shape": (3,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([0, 2, 4])),
        "hint": "Multiply by 2",
    },
    # 34. Reshape to 3D
    {
        "input": lambda: np.arange(8),
        "target": lambda: np.arange(8).reshape(2, 2, 2),
        "target_shape": (2, 2, 2),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.arange(8).reshape(2, 2, 2)),
        "hint": "Use .reshape(2, 2, 2)",
    },
    # 35. Flip axis
    {
        "input": lambda: np.arange(6).reshape(2, 3),
        "target": lambda: np.array([[2, 1, 0], [5, 4, 3]]),
        "target_shape": (2, 3),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([[2, 1, 0], [5, 4, 3]])),
        "hint": "Use np.fliplr(x) or x[:, ::-1]",
    },
    # 36. Stack along new axis
    {
        "input": lambda: np.array([1, 2, 3]),
        "target": lambda: np.array([[1, 2, 3], [1, 2, 3]]),
        "target_shape": (2, 3),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([[1, 2, 3], [1, 2, 3]])),
        "hint": "Use np.stack([x, x])",
    },
    # 37. Remove axis
    {
        "input": lambda: np.ones((1, 1, 5)),
        "target": lambda: np.ones((5,)),
        "target_shape": (5,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.ones((5,))),
        "hint": "Use x.squeeze()",
    },
    # 38. Insert axis
    {
        "input": lambda: np.array([1, 2, 3]),
        "target": lambda: np.array([[[1, 2, 3]]]),
        "target_shape": (1, 1, 3),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([[[1, 2, 3]]])),
        "hint": "Use x[None, None, :]",
    },
    # 39. Argwhere
    {
        "input": lambda: np.array([0, 1, 0, 2]),
        "target": lambda: np.array([[1], [3]]),
        "target_shape": (2, 1),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([[1], [3]])),
        "hint": "Use np.argwhere(x)",
    },
    # 40. Unique
    {
        "input": lambda: np.array([1, 2, 2, 3, 1]),
        "target": lambda: np.array([1, 2, 3]),
        "target_shape": (3,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([1, 2, 3])),
        "hint": "Use np.unique(x)",
    },
    # 41. Sort
    {
        "input": lambda: np.array([3, 1, 2]),
        "target": lambda: np.array([1, 2, 3]),
        "target_shape": (3,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([1, 2, 3])),
        "hint": "Use np.sort(x)",
    },
    # 42. Argsort
    {
        "input": lambda: np.array([3, 1, 2]),
        "target": lambda: np.array([1, 2, 0]),
        "target_shape": (3,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([1, 2, 0])),
        "hint": "Use x.argsort()",
    },
    # 43. Cumulative product
    {
        "input": lambda: np.array([1, 2, 3, 4]),
        "target": lambda: np.array([1, 2, 6, 24]),
        "target_shape": (4,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([1, 2, 6, 24])),
        "hint": "Use x.cumprod()",
    },
    # 44. Diagonal extraction
    {
        "input": lambda: np.arange(9).reshape(3, 3),
        "target": lambda: np.array([0, 4, 8]),
        "target_shape": (3,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([0, 4, 8])),
        "hint": "Use np.diag(x)",
    },
    # 45. Trace
    {
        "input": lambda: np.arange(9).reshape(3, 3),
        "target": lambda: np.array(12),
        "target_shape": (),
        "test": lambda r: np.isscalar(r) and r == 12,
        "hint": "Use np.trace(x)",
    },
    # 46. Reshape and sum
    {
        "input": lambda: np.arange(12),
        "target": lambda: np.array([6, 22]),
        "target_shape": (2,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([6, 22])),
        "hint": "Use x.reshape(2, 6).sum(axis=1)",
    },
    # 47. Masked sum
    {
        "input": lambda: np.arange(10),
        "target": lambda: np.array(30),
        "target_shape": (),
        "test": lambda r: np.isscalar(r) and r == 30,
        "hint": "Sum only even numbers: x[x % 2 == 0].sum()",
    },
    # 48. Boolean indexing
    {
        "input": lambda: np.array([[1, 2], [3, 4]]),
        "target": lambda: np.array([2, 4]),
        "target_shape": (2,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([2, 4])),
        "hint": "x[x > 1]",
    },
    # 49. Advanced indexing
    {
        "input": lambda: np.array([10, 20, 30, 40]),
        "target": lambda: np.array([40, 10]),
        "target_shape": (2,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([40, 10])),
        "hint": "x[[3, 0]]",
    },
    # 50. Hard: 3D transpose and sum
    {
        "input": lambda: np.arange(24).reshape(2, 3, 4),
        "target": lambda: np.array([66, 210]),
        "target_shape": (2,),
        "test": lambda r: isinstance(r, np.ndarray)
        and np.array_equal(r, np.array([66, 210])),
        "hint": "x.transpose(2, 1, 0).sum(axis=(0, 1))",
    },
]
