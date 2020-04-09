'''
代理模式，让一些计算开销比较大的属性，延迟初始化。
比如：Adaptive 中的属性dict
'''
class LazyProperty:

    def __init__(self, method):
        self.method = method
        self.method_name = method.__name__
        # print('function overriden: {}'.format(self.method))
        # print("function's name: {}".format(self.method_name))

    def __get__(self, obj, cls):
        if not obj:
            return None
        value = self.method(obj)
        # print('value {}'.format(value))
        setattr(obj, self.method_name, value)
        return value