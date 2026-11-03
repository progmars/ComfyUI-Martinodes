from .shared import get_batch_size, select_batch_item, unwrap_singleton_batch_input, CATEGORY


class BatchBoundPicker:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_batch": ("*",),
                "mode": (["first", "last"],),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("selected",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Selects the first or last item from a batch."


    def run(self, any_batch, mode):
        if any_batch is None:
            return (None,)
        any_batch = unwrap_singleton_batch_input(any_batch)
        mode = mode[0] if isinstance(mode, list) else mode
        batch_size = get_batch_size(any_batch)
        if batch_size == 0:
            return (None,)

        index = 0 if mode == "first" else batch_size - 1
        return (select_batch_item(any_batch, index, batch_size),)
