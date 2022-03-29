class WrapOp:
    '''
    init固定两个参数 op 和 apply
    - op固定是init第一个参数。一个函数，和现在上面的这些实现一样，针对一个3D数组写
    - apply可以用两种方式传
        1. WrapOp(op, apply="image")
        2. WrapOp(op, "both") 第二个匿名参数
        第一种优先级高，如果有apply=''的参数了第二个匿名参数就被认为是op参数。op是针对一个，3d array的，应该也不会有值是image/label/both的情况

    apply 默认值both。apply=
    - both：image,label都做op
    - image：只image做
    - label：只label做

    其他的所有参数都是op的参数，参数值的类型分两类：tuple和其他。
    - 如果想image和label用不同的参数，参数用长度 == 2的tuple。比如resize，image是3阶，label是0阶：WrapOp(op, "both", order=(3, 0))
    - 如果image和label参数一样，用其他的类型。比如都用0阶，大小都是128^3：WrapOp(op, "both", order=0, size=[128, 128, 128])

    这样有一个限制，op的参数不能限定必须用tuple，那样在WrapOp层这个参数就会被拆开。不过感觉问题不大，能用tuple的地方应该都差不多能用list传参
    '''

    def __init__(self, op, *args, **kwargs):
        print("====")
        self.op = op
        self.apply = None
        # 先从kwargs里找
        if 'apply' in kwargs.keys():
            self.apply = kwargs['apply']
            del kwargs['apply']
        # 之后看args第一个
        if self.apply is None and len(args) and args[0] in ["both", "image", "label"]:
            self.apply = args[0]
            args = list(args)
            del args[0]
        # 不传默认 both
        if self.apply is None:
            self.apply = "both"
        print("self.apply", self.apply)
        print("args: ", args, "kwargs: ", kwargs)
        self.image_args = []
        self.label_args = []
        self.image_kwargs = {}
        self.label_kwargs = {}
        for idx, target_args in enumerate([self.image_args, self.label_args]):
            for arg in args:
                # tuple类型的参数第一个给image，第二个给label。否则image和label的这个参数一样
                if isinstance(arg, tuple):
                    assert len(arg) == 2, f"The length of tuple type input must be 2, got {arg}"
                    target_args.append(arg[idx])
                else:
                    target_args.append(arg)

        for idx, target_kwargs in enumerate([self.image_kwargs, self.label_kwargs]):
            for k, v in kwargs.items():
                if isinstance(v, tuple):
                    target_kwargs[k] = v[idx]
                else:
                    target_kwargs[k] = v
        print(self.image_args, self.label_args, self.image_kwargs, self.label_kwargs)

    '''
    在load_save里调 prep_op.run(image, label)。如果图像和标签的处理需要通信重载这个方法，否则直接用这个run就行
    '''
    def run(self, image, label):
        if self.apply in ("both", "image"):
            image = self.op(image, *self.image_args, **self.image_kwargs)

        if self.apply in ("both", "label"):
            label = self.op(label, *self.label_args, **self.label_kwargs)
        return image, label

if __name__ == "__main__":

    def add(image, val):
        return image + val

    def sub(image, val):
        return image - val

    class CrossOp(WrapOp):
        def run(self, image, label):
            return image * label, image / label

    ops = [WrapOp(add, "both", 1), WrapOp(sub, (2, 3)), WrapOp(sub, "label", 4), WrapOp(add, 4, apply="image"), CrossOp(None)]
    print("finish create, start run")
    image = 0
    label = 0
    for op in ops:
        image, label = op.run(image, label)
        print(image, label)
