import numpy as np
import json
from levels import levels

if "current_level" not in globals():
    current_level = 0


def get_level_data():
    level = levels[current_level]
    x = level["input"]()
    target = level["target"]()
    return json.dumps({
        "level": current_level + 1,
        "input_tensor": f"x = {repr(x)}",
        "input_tensor_str": np.array2string(x, separator=', '),
        "target_tensor": f"{repr(target)} (shape: {target.shape})",
        "target_tensor_str": np.array2string(target, separator=', '),
        "code_default": "",
        "hint": level["hint"]
    })


def check_user_code(user_code):
    level = levels[current_level]
    x = level["input"]()
    try:
        local_vars = {"x": x.copy(), "np": np}
        exec(f"result = {user_code}", {}, local_vars)
        result = local_vars["result"]
        result_str = np.array2string(result, separator=', ') if isinstance(result, np.ndarray) else str(result)
        if level["test"](result):
            msg = f"✅ Correct! Result: {repr(result)}"
            if current_level + 1 >= len(levels):
                msg += "\n🎉 All levels completed!"
            return json.dumps({"result": "correct", "message": msg, "user_result": result_str})
        else:
            return json.dumps({"result": "incorrect", "message": f"❌ Incorrect! Got: {repr(result)} (shape: {getattr(result, 'shape', None)})", "user_result": result_str})
    except Exception as e:
        return json.dumps({"result": "error", "message": f"❌ Error: {str(e)}", "user_result": str(e)})


def next_level():
    global current_level
    current_level += 1
    if current_level >= len(levels):
        current_level = 0
