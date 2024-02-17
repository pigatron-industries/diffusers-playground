import copy


class DeepCopyObject:

    def __init__(self):
        self.deepcopy_excluded_modules = []


    def __deepcopy__(self, memo):
        # print(f"Copying object: {self}")
        if(self.deepcopy_excluded_modules is None):
            self.deepcopy_excluded_modules = []
        copied_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied_obj
        for key, value in self.__dict__.items():
            if any(value.__class__.__module__.startswith(module) for module in self.deepcopy_excluded_modules):
                continue
            # print(f"Copying attribute: {key}")
            setattr(copied_obj, key, copy.deepcopy(value, memo))
        return copied_obj