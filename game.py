import json

import numpy as np

from levels import levels

# Simple global state dictionary - should persist in Pyodide
_game_state = {"current_level": 0, "failed_attempts": 0}


# Use a class to maintain state in Pyodide
class GameState:
    def __init__(self):
        self.current_level = 0
        self.failed_attempts = 0


# Create global game state instance
game_state = GameState()


def get_level_data():
    level = levels[_game_state["current_level"]]
    x = level["input"]()
    target = level["target"]()
    show_hint = _game_state["failed_attempts"] >= 3
    print(
        f"DEBUG: Level {_game_state['current_level']}, Failed attempts: {_game_state['failed_attempts']}, Show hint: {show_hint}"
    )
    return json.dumps(
        {
            "level": _game_state["current_level"] + 1,
            "input_tensor": f"x = {repr(x)}",
            "input_tensor_str": np.array2string(x, separator=", "),
            "target_tensor": f"{repr(target)} (shape: {target.shape})",
            "target_tensor_str": np.array2string(target, separator=", "),
            "code_default": "",
            "hint": level["hint"] if show_hint else "",
            "show_hint": show_hint,
            "failed_attempts": _game_state["failed_attempts"],
        }
    )


def check_user_code(user_code):
    level = levels[_game_state["current_level"]]
    x = level["input"]()
    try:
        local_vars = {"x": x.copy(), "np": np}
        exec(f"result = {user_code}", {}, local_vars)
        result = local_vars["result"]
        result_str = (
            np.array2string(result, separator=", ")
            if isinstance(result, np.ndarray)
            else str(result)
        )
        if level["test"](result):
            msg = f"✅ Correct! Result: {repr(result)}"
            if _game_state["current_level"] + 1 >= len(levels):
                msg += "\n🎉 All levels completed!"
            _game_state["failed_attempts"] = 0  # Reset on success
            return json.dumps(
                {
                    "result": "correct",
                    "message": msg,
                    "user_result": result_str,
                }
            )
        else:
            _game_state["failed_attempts"] += 1
            print(
                f"DEBUG: Failed attempts now: {_game_state['failed_attempts']}"
            )
            return json.dumps(
                {
                    "result": "incorrect",
                    "message": f"❌ Incorrect! Got: {repr(result)} (shape: {getattr(result, 'shape', None)})",
                    "user_result": result_str,
                }
            )
    except Exception as e:
        _game_state["failed_attempts"] += 1
        print(f"DEBUG: Failed attempts now: {_game_state['failed_attempts']}")
        return json.dumps(
            {
                "result": "error",
                "message": f"❌ Error: {str(e)}",
                "user_result": str(e),
            }
        )


def get_failed_attempts():
    return _game_state["failed_attempts"]


def next_level():
    _game_state["current_level"] += 1
    _game_state["failed_attempts"] = 0  # Reset failed attempts for new level
    if _game_state["current_level"] >= len(levels):
        _game_state["current_level"] = 0


def debug_state():
    """Debug function to check current game state"""
    return json.dumps(
        {
            "current_level": _game_state["current_level"],
            "failed_attempts": _game_state["failed_attempts"],
            "game_state_id": id(_game_state),
        }
    )


# Alternative: Store state in Pyodide globals as backup
def init_game_globals():
    """Initialize game state in Pyodide globals"""
    import sys

    if not hasattr(sys.modules[__name__], "_game_initialized"):
        sys.modules[__name__]._current_level = 0
        sys.modules[__name__]._failed_attempts = 0
        sys.modules[__name__]._game_initialized = True


def get_global_state():
    """Get state from module globals"""
    import sys

    module = sys.modules[__name__]
    return {
        "current_level": getattr(module, "_current_level", 0),
        "failed_attempts": getattr(module, "_failed_attempts", 0),
    }


def set_global_state(current_level=None, failed_attempts=None):
    """Set state in module globals"""
    import sys

    module = sys.modules[__name__]
    if current_level is not None:
        module._current_level = current_level
    if failed_attempts is not None:
        module._failed_attempts = failed_attempts


# Initialize on module load
init_game_globals()
