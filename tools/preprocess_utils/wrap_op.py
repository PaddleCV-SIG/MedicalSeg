class WrapOp:
    '''
    支持类似函数式或subclass两种使用方法，同时处理图像和分割标签

    init固定两个参数 op 和 apply
    - op针对一个3d volume写，如果用subclass方式传None
    - apply 默认值both。apply=
        - both：image，label都做op
        - image：只image做
        - label：只label做

    其他的所有参数都是op的参数，必须写参数名，参数值的类型分两类：tuple和其他。
    - 如果想image和label用不同的参数，参数用长度 == 2 的tuple。比如resize，image是3阶，label是0阶：WrapOp(op, "both", order=(3, 0))
    - 如果image和label参数一样，用其他的类型。比如都用0阶，大小都是128^3：WrapOp(op, "both", order=0, size=[128, 128, 128])

    这样有一个限制，op的参数不能限定必须用tuple，那样在WrapOp层这个参数就会被拆开。不过感觉问题不大，能用tuple的地方应该都差不多能用list传参
    '''

    def __init__(self, op, apply="both", **kwargs):
        print("====")
        if op is not None:
            self.op = op
        self.apply = apply

        self.image_kwargs = {}
        self.label_kwargs = {}
        for idx, target_kwargs in enumerate([self.image_kwargs, self.label_kwargs]):
            for k, v in kwargs.items():
                if isinstance(v, tuple):
                    target_kwargs[k] = v[idx]
                else:
                    target_kwargs[k] = v

        print("self.apply", self.apply)
        print("kwargs: ", kwargs)
        print(self.image_kwargs, self.label_kwargs)


    '''
    在load_save里调 prep_op.run(image, label)。如果图像和标签的处理需要通信重载这个方法，否则直接用这个run就行
    '''
    def run(self, image, label):
        if self.apply in ("both", "image"):
            image = self.op(image, **self.image_kwargs)

        if self.apply in ("both", "label"):
            label = self.op(label, **self.label_kwargs)
        return image, label

if __name__ == "__main__":

    def add(image, val):
        return image + val

    class Sub(WrapOp):
        def __init__(self, apply="both", **kwargs):
            super().__init__(None, apply=apply, **kwargs)

        def op(self, image, val):
            return image - val

    class CrossOp(WrapOp):
        def __init__(self):
            super().__init__(None, apply="both")

        def run(self, image, label):
            return image * label, image / label

    ops = [
        WrapOp(add, "both", val=1),
        Sub(val=(2, 3)),
        Sub("label", val=4),
        WrapOp(add, apply="image", val=4),
        CrossOp()
    ]
    print("finish create, start run")

    image = 0
    label = 0
    for op in ops:
        image, label = op.run(image, label)
        print(image, label)
