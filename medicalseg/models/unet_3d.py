import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from medicalseg.cvlibs import manager

class ConvActDropNorm(nn.Sequential):
    def __init__(self,input_channels,outchannel,kernel_size,stride,padding):
        super().__init__()
        self.add_sublayer("conv",nn.Conv3D(input_channels,outchannel,kernel_size=kernel_size,stride=stride,padding=padding))
        self.add_sublayer("bn",nn.BatchNorm3D(outchannel))
        self.add_sublayer("prelu",nn.PReLU())


class EncoderLayer(nn.Layer):
    def __init__(self,input_channels,out_channels,path_length=1):
        super().__init__()
        self.path_length=path_length
        self.conv1=ConvActDropNorm(input_channels,out_channels,kernel_size=3,stride=1,padding=1)
        path_convs=[ConvActDropNorm(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
                                    for i in range(self.path_length)]
        self.path_convs=nn.LayerList(path_convs)
        self.downSample=ConvActDropNorm(out_channels,out_channels,kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        output=self.conv1(x)
        for conv in self.path_convs:
            output=conv(output)
        x=self.downSample(output)
        return output,x

class Encoder(nn.Layer):
    def __init__(self,input_channels,channles,init_channels=32):
        super().__init__()
        # self.input_channels=input_channels
        layer_list=[]
        for channel in channles[:-1]:#[32,64,128,]
            layer_list.append(EncoderLayer(input_channels,channel))
            input_channels=channel
        self.encoders=nn.LayerList(layer_list)
        self.bottom_conv=ConvActDropNorm(channles[-2],channles[-1],kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        output_list=[]
        for encoder in self.encoders:
            #output 2 values: output and output after downsampling
            output,x=encoder(x)
            #       residual                 x
            # [batch,32,128,128,128]、[batch,32,64,64,64]=>
            # [batch,64,64,64,64]、[batch,64,32,32,32]=>
            # [batch,128,32,32,32]、[batch,128,16,16,16]=>[batch,256,16,16,16]

            output_list.append(output)
        x=self.bottom_conv(x)
        return output_list,x

class DecoderLayer(nn.Layer):
    def __init__(self,input_channel,out_channel,path_length=1):
        super().__init__()
        self.path_length = path_length
        self.upsample=nn.Conv3DTranspose(input_channel,out_channel,kernel_size=2,stride=2,padding=0)
        path_convs=[ConvActDropNorm(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
                                        for i in range(self.path_length)]
        self.path_convs = nn.LayerList(path_convs)
    def forward(self,x,residual):
        x=self.upsample(x)
        x=x+residual
        for conv in self.path_convs:
            x=conv(x)
        return x

class Decoder(nn.Layer):
    def __init__(self,n_classes,channels):
        super().__init__()
        layer_list=[]
        #channels [32,64,128,256])
        for i in range(len(channels)-1,0,-1):
            layer_list.append(DecoderLayer(channels[i],channels[i-1]))
        self.decoders=nn.LayerList(layer_list)
        self.top_conv=ConvActDropNorm(channels[0],n_classes,kernel_size=1,stride=1,padding=0)

    def forward(self, residual_list,x):
        for decoder in self.decoders:
            residual=residual_list.pop()
            x=decoder(x,residual)
        x=self.top_conv(x)
        return x

@manager.MODELS.add_component
class Unet3d(nn.Layer):
    def __init__(self,input_channels,n_classes,channles=None,downSample='maxpool',nll=False):
        super().__init__()
        #channels [32,64,128,256]
        self.input_channels=input_channels
        self.n_classes=n_classes
        self.channels=channles
        self.encoder=Encoder(input_channels,self.channels)
        self.decoder=Decoder(n_classes,self.channels)
        if nll:
            self.softmax=F.log_softmax
        else:
            self.softmax=F.softmax

    def forward(self,x):#[batchsize,1,128,128,128]
        output_residual,out=self.encoder(x)
        output=self.decoder(output_residual,out)
        return self.softmax(output,axis=1)

if __name__=='__main__':
    input_channels=1
    n_classes=2

    unet=Unet3d(input_channels,n_classes,[32,64,128,256])
    unet.eval()
    for i in range(100):
        input = paddle.rand([1, input_channels, 128, 128, 128])
        output=unet(input)
        print(output.shape)
