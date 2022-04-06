class WrapOp:
    '''
    包装预处理函数，支持同时处理图像和标签(apply=together)

    init固定两个参数 op 和 apply
    - op：预处理函数。apply=image，label，both的时候op针对一个3d volume写；apply=together的时候image和volume会一起传给op
    - apply：默认值both。apply=
        - both：image，label分别做op，图像和标签处理过程中不共享数据
        - image：只image做
        - label：只label做
        - together：op同时处理image和label

    其他的所有参数都是op的参数，必须写参数名，参数值的类型分两类：tuple和其他。
    - tuple：image和label用不同的参数，参数用长度为2的tuple。比如resize操作，image是3阶，label是0阶：WrapOp(resize, "both", order=(3, 0))
    - 其它：如果image和label参数一样，用其他的类型。比如resize都用0阶，大小都是128^3：WrapOp(resize, "both", order=0, size=[128, 128, 128])
    '''

    def __init__(self, op, apply="both", **params_dict):
        self.op = op
        self.apply = apply

        self.image_params = {}
        self.label_params = {}
        self.update_params(params_dict)

    def update_params(self, params_dict):
        for idx, target_params in enumerate([self.image_params, self.label_params]):
            for k, v in params_dict.items():
                if k == 'apply':
                    continue
                if isinstance(v, tuple):
                    assert len(v) == 2, f"Parameters of type tuple must have length 2, got parameter {k} with value {v}"
                    target_params[k] = v[idx]
                else:
                    target_params[k] = v

    '''
    在load_save里调 prep_op.run(image, label)
    '''
    def run(self, image, label, **params_dict):
        self.update_params(params_dict)
        if self.apply == "together":
            return self.op(image, label, self.image_params, self.label_params)
        
        if self.apply in ("both", "image"):
            image = self.op(image, **self.image_params)

        if self.apply in ("both", "label"):
            label = self.op(label, **self.label_params)

        return image, label

# if __name__ == "__main__":

#     def add(image, val):
#         return image + val

#     def sub(image, val):
#         return image - val
    
#     def add_mul(image, label, image_param, label_param):
#         image += label
#         label = image
#         image *= image_param["mul"]
#         label *= label_param["mul"]
#         return image, label

#     ops = [
#         WrapOp(add, "both", val=1),
#         WrapOp(add, "both", val=(2, 3)),
#         WrapOp(sub, "label", val=2),
#         WrapOp(add, apply="image", val=4),
#         WrapOp(add_mul, apply="together")
#     ]
#     print("finish create, start run")

#     image = 0
#     label = 0
#     for op in ops:
#         if op.op == add_mul:
#             image, label = op.run(image, label, mul=(2, 3))    
#         else:
#             image, label = op.run(image, label)
#         print(image, label)
