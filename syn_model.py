import torch
import torch.nn as nn
from model.non_local_embedded_gaussian import NONLocalBlock2D

def up(x): 
    return nn.functional.interpolate(x,scale_factor=2)
        
def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def conv_decod_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


class Non_Local_Block_0(nn.Module):
    
    def __init__(self,in_dim):
        super(Non_Local_Block_0, self).__init__()
        self.non_local = NONLocalBlock2D(in_channels=in_dim)

    def forward(self, x1):

        fusion_feature = self.non_local(x1)
       
        return fusion_feature


class Residual_Block_0(nn.Module):
    def __init__(self,in_dim,act_fn):
        super(Residual_Block_0, self).__init__()
        self.layer1 = nn.Sequential(Non_Local_Block_0(in_dim),act_fn)
        self.layer2 = nn.Sequential(Non_Local_Block_0(in_dim))


    def forward(self, x):
        
        
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + x
         
        return out2
    

class Residual_Block(nn.Module):
    def __init__(self,in_dim,act_fn):
        super(Residual_Block, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),act_fn)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim))


    def forward(self, x):
        
        
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + x
         
        return out2

class MixedFusion_Block(nn.Module):
    
    def __init__(self,in_dim, out_dim,act_fn):
        super(MixedFusion_Block, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),act_fn,)
        
        # revised in 09/09/2019.
        #self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim,  kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)


    def forward(self, x1,x2,xx):
        
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)   # sum
        fusion_mul = torch.mul(x1, x2)
         
        modal_in1  = torch.reshape(x1,[x1.shape[0],1,x1.shape[1],x1.shape[2],x1.shape[3]])
        modal_in2  = torch.reshape(x2,[x2.shape[0],1,x2.shape[1],x2.shape[2],x2.shape[3]])
        modal_cat  = torch.cat((modal_in1, modal_in2),dim=1)
        fusion_max = modal_cat.max(dim=1)[0]
         
        out_fusion = torch.cat((fusion_sum,fusion_mul,fusion_max),dim=1)
        
        out1 = self.layer1(out_fusion)
        out2 = self.layer2(torch.cat((out1,xx),dim=1))
        
        return out2
        

class MixedFusion_Block0(nn.Module):
    def __init__(self,in_dim, out_dim,act_fn):
        super(MixedFusion_Block0, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),act_fn,)
        #self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)


    def forward(self, x1,x2):
        
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)   # sum
        fusion_mul = torch.mul(x1, x2)
         
        modal_in1  = torch.reshape(x1,[x1.shape[0],1,x1.shape[1],x1.shape[2],x1.shape[3]])
        modal_in2  = torch.reshape(x2,[x2.shape[0],1,x2.shape[1],x2.shape[2],x2.shape[3]])
        modal_cat  = torch.cat((modal_in1, modal_in2),dim=1)
        fusion_max = modal_cat.max(dim=1)[0]
         
        out_fusion = torch.cat((fusion_sum,fusion_mul,fusion_max),dim=1)
        out_softmax = nn.functional.softmax(out_fusion)
        out_fusion = out_fusion * out_softmax
        out1 = self.layer1(out_fusion)
        out2 = self.layer2(out1)
         
        return out2



##############################################
# define our model
class Multi_modal_generator(nn.Module):

    def __init__(self,input_nc, output_nc, ngf):
        super(Multi_modal_generator,self).__init__()
        
        
        self.in_dim  = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        #act_fn = nn.ReLU()
        
        act_fn2 = nn.ReLU(inplace=True) #nn.ReLU()

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1)
        
        #######################################################################
        # Encoder **Modality 1
        #######################################################################
        self.down_1_0 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_dim,  out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim), act_fn,
                nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim), act_fn,
                )#224*224*32
        self.pool_1_0 = maxpool()#112*112*32
        
        self.down_2_0 = nn.Sequential(
                nn.Conv2d(in_channels=self.out_dim,  out_channels=self.out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*2), act_fn,
                nn.Conv2d(in_channels=self.out_dim*2,out_channels=self.out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*2), act_fn,
                )#112*112*64
        self.pool_2_0 = maxpool()#56*56*64
        
        self.down_3_0 = nn.Sequential(
                nn.Conv2d(in_channels=self.out_dim*2,  out_channels=self.out_dim*4, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*4), act_fn,
                nn.Conv2d(in_channels=self.out_dim*4,out_channels=self.out_dim*4, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*4), act_fn,
                )#56*56*128
        self.pool_3_0 = maxpool()#28*28*128

        
        #######################################################################
        # Encoder **Modality 2
        #######################################################################
        self.down_1_1 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_dim,  out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim), act_fn,
                nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim), act_fn,
                )
        self.pool_1_1 = maxpool()
        
        self.down_2_1 = nn.Sequential(
                nn.Conv2d(in_channels=self.out_dim,  out_channels=self.out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*2), act_fn,
                nn.Conv2d(in_channels=self.out_dim*2,out_channels=self.out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*2), act_fn,
                )
        self.pool_2_1 = maxpool()
        
        self.down_3_1 = nn.Sequential(
                nn.Conv2d(in_channels=self.out_dim*2,  out_channels=self.out_dim*4, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*4), act_fn,
                nn.Conv2d(in_channels=self.out_dim*4,out_channels=self.out_dim*4, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*4), act_fn,
                )
        self.pool_3_1 = maxpool()


         
        #######################################################################
        # bottleneck layer
        #######################################################################

         #bottleneck layer 1
        self.bl_1_1 = Residual_Block_0(self.out_dim * 4,act_fn)
        self.bl_1_2 = Residual_Block(self.out_dim * 2,act_fn)
        self.bl_1_3 = Residual_Block(self.out_dim * 1,act_fn)

        #bottleneck layer 2
        self.bl_2_1 = Residual_Block_0(self.out_dim * 4,act_fn)
        self.bl_2_2 = Residual_Block(self.out_dim * 2,act_fn)
        self.bl_2_3 = Residual_Block(self.out_dim * 1,act_fn)
        
        # Modality 1
        self.deconv_1_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)#28*28*128
        self.deconv_2_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)#56*56*128
        self.deconv_3_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)#56*56*64 ---56*56*128
        self.deconv_4_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)#112*112*64 ----112*112*64
        self.deconv_5_1 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn2)#112*112*32 ----112*112*64
        self.deconv_6_1 = conv_decod_block(self.out_dim * 2, int(self.out_dim), act_fn2)#224*224*32
        self.mask1      = nn.Sequential(nn.Conv2d(int(self.out_dim),1, kernel_size=3, stride=1, padding=1))
        self.out1       = nn.Sequential(nn.Conv2d(int(self.out_dim),1, kernel_size=3, stride=1, padding=1),nn.Tanh()) #  self.final_out_dim #224*224*1
        
        # modality 2
        self.deconv_1_2 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)#28*28*128
        self.deconv_2_2 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)#56*56*128        
        self.deconv_3_2 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)#56*56*64
        self.deconv_4_2 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)#112*112*64
        self.deconv_5_2 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn2)#112*112*32
        self.deconv_6_2 = conv_decod_block(self.out_dim * 2, int(self.out_dim), act_fn2)#224*224*32
        self.mask2      = nn.Sequential(nn.Conv2d(int(self.out_dim),1, kernel_size=3, stride=1, padding=1))
        self.out2       = nn.Sequential(nn.Conv2d(int(self.out_dim),1, kernel_size=3, stride=1, padding=1),nn.Tanh()) #  self.final_out_dim #224*224*1


        # Encoder **Modality 3
        #######################################################################
        self.down_1_3 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_dim,  out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim), act_fn,
                nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim), act_fn,
                )
        self.pool_1_3 = maxpool()


        #######################################################################
        # fusion layer
        #######################################################################
        # down 1st layer
        self.down_fu_1 = MixedFusion_Block(self.out_dim,self.out_dim*2,act_fn)
        self.pool_fu_1 = maxpool()
        
        self.down_fu_2 = MixedFusion_Block(self.out_dim*2,self.out_dim*4,act_fn)
        self.pool_fu_2 = maxpool()
        
        self.down_fu_3 = MixedFusion_Block(self.out_dim*4,self.out_dim*4,act_fn)
        #self.pool_fu_3 = maxpool()

        # down 4th layer
        #self.down_fu_4 = nn.Sequential(nn.Conv2d(in_channels=self.out_dim*4,  out_channels=self.out_dim*8, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*8), act_fn,)   
        #bottleneck layer ts
        self.bl_ts_1 = Residual_Block_0(self.out_dim * 4,act_fn)
    

        # ~~~ Decoding Path ~~~~~~ #
        self.deconv_1_0 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.deconv_2_0 = MixedFusion_Block(self.out_dim * 4, self.out_dim * 2,act_fn2)
        self.deconv_3_0 = MixedFusion_Block(self.out_dim * 2, self.out_dim * 1,act_fn2)
        self.deconv_4_0 = MixedFusion_Block(self.out_dim * 1, self.out_dim,act_fn2)  
        self.deconv_5_0 = conv_decod_block(self.out_dim * 1, self.out_dim,act_fn2) 
        self.out        = nn.Sequential(nn.Conv2d(int(self.out_dim),1, kernel_size=3, stride=1, padding=1),nn.Tanh()) #  self.final_out_dim

    
                
    def forward(self,inputs):

        # ############################# #
        i0 = inputs[:,0:1,:,:]
        i1 = inputs[:,1:2,:,:]
        
        # -----  First Level -------- 
        down_1_0 = self.down_1_0(i0) 
        down_1_1 = self.down_1_1(i1)#224*224*32 


        # -----  Second Level --------
        #input_2nd = torch.cat((down_1_0,down_1_1,down_1_2,down_1_3),dim=1)
        # Max-pool
        down_1_0m   = self.pool_1_0(down_1_0)
        down_1_1m   = self.pool_1_1(down_1_1)#112*112*32
        
        down_2_0 = self.down_2_0(down_1_0m)
        down_2_1 = self.down_2_1(down_1_1m)#112*112*64
        
        
        # -----  Third Level --------
        # Max-pool
        down_2_0m = self.pool_2_0(down_2_0)
        down_2_1m = self.pool_2_1(down_2_1)#56*56*64
                
        down_3_0 = self.down_3_0(down_2_0m)
        down_3_1 = self.down_3_1(down_2_1m)#56*56*128
        
        # Max-pool
        down_3_0m = self.pool_3_0(down_3_0)
        down_3_1m = self.pool_3_1(down_3_1)#28*28*128
         
        # ----------------------------------------
        # bottleneck layer 1
        down_3_0m   = self.bl_1_1(down_3_0m)                                                                                                         
        down_2_0m  = self.bl_1_2(down_2_0m)  
        down_1_0m   = self.bl_1_3(down_1_0m)                                                                                                         

        # bottleneck layer 2
        down_3_1m   = self.bl_2_1(down_3_1m)                                                                                                         
        down_2_1m  = self.bl_2_2(down_2_1m)  
        down_1_1m   = self.bl_2_3(down_1_1m)
       

        #######################################################################                                                                                                

        # modality 1
        deconv_1_1 = self.deconv_1_1((down_3_0m))#28*28*128
        deconv_2_1 = self.deconv_2_1(up(deconv_1_1))#56*56*128
        deconv_3_1_ = self.deconv_3_1((deconv_2_1))#56*56*64
        deconv_3_1 = torch.cat((deconv_3_1_,down_2_0m),dim=1)
        deconv_4_1 = self.deconv_4_1(up(deconv_3_1))#112*112*64
        deconv_5_1_ = self.deconv_5_1((deconv_4_1))#112*112*32
        deconv_5_1 = torch.cat((deconv_5_1_,down_1_0m),dim=1)
        deconv_6_1 = self.deconv_6_1(up(deconv_5_1))#224*224*32
        output1   = self.out1(deconv_6_1)#224*224*2
        mask_1    = self.mask1(deconv_6_1)
        
        # modality 2
        deconv_1_2 = self.deconv_1_2((down_3_1m))
        deconv_2_2 = self.deconv_2_2(up(deconv_1_2))
        deconv_3_2_ = self.deconv_3_2((deconv_2_2))
        deconv_3_2 = torch.cat((deconv_3_2_,down_2_1m),dim=1)
        deconv_4_2 = self.deconv_4_2(up(deconv_3_2))
        deconv_5_2_ = self.deconv_5_2((deconv_4_2))
        deconv_5_2 = torch.cat((deconv_5_2_,down_1_1m),dim=1)
        deconv_6_2 = self.deconv_6_2(up(deconv_5_2))
        output2    = self.out2(deconv_6_2)
        mask_2    = self.mask2(deconv_6_2)

        #confidence aggregatation
        all_mask = torch.cat((mask_1,mask_2),dim=1)
        all_mask = nn.functional.softmax(all_mask,dim=1)
        output_1 = torch.unsqueeze(output1,dim=1)
        output_2 = torch.unsqueeze(output2,dim=1)
        all_image = torch.cat((output_1,output_2),dim=1)
        all_mask_e = torch.unsqueeze(all_mask,dim=2)
        output = torch.sum(all_mask_e * all_image,dim=1)
        #output1_softmax = all_mask[:,0:1,:,:] * output1
        #output2_softmax = all_mask[:,1:2,:,:] * output2
        #output = output1_softmax + output2_softmax
        mask_softmax_1 = all_mask[:,0,:,:]
        mask_softmax_2 = all_mask[:,1,:,:]

        # ----------------------------------------
        # synthesized modality 3
        # -----  First Level -------- 
        down_1_3 = self.down_1_3(output) 

        # Max-pool
        down_1_3m   = self.pool_1_3(down_1_3)

        # bottleneck layer 1
        #down_3_3m   = self.bl_1_1(down_3_3m)
        
        # feature fusion layer
        down_fu_1   = self.down_fu_1(down_1_0m,down_1_1m,down_1_3m)                                                                                                         
        down_fu_1m  = self.pool_fu_1(down_fu_1)

        down_fu_2   = self.down_fu_2(down_2_0m,down_2_1m,down_fu_1m)                                                                                                         
        down_fu_2m  = self.pool_fu_2(down_fu_2)

        down_fu_3   = self.down_fu_3(down_3_0m,down_3_1m,down_fu_2m) 

        # bottleneck layer 1
        down_fu_4   = self.bl_ts_1(down_fu_3)#28*28

        # ~~~~~~ Decoding 
        deconv_1_0 = self.deconv_1_0(down_fu_4)#28*28
        deconv_2_0 = self.deconv_2_0(deconv_1_1,deconv_1_2,deconv_1_0)#28*28
        deconv_3_0 = self.deconv_3_0(deconv_3_1_,deconv_3_2_,up(deconv_2_0))#56*56
        deconv_4_0 = self.deconv_4_0(deconv_5_1_,deconv_5_2_,up(deconv_3_0))#112*112
        deconv_5_0 = self.deconv_5_0(up(deconv_4_0))#224*224
        output_ts     = self.out(deconv_5_0)#224*224
                 
        return output_ts,output,output1,output2,mask_softmax_1,mask_softmax_2

  
  
 

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discrimintor_block(in_features, out_features, normalize=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discrimintor_block(in_channels, 32, normalize=False),
            *discrimintor_block(32, 64),
            *discrimintor_block(64, 128),
            *discrimintor_block(128, 256),
            #nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, kernel_size=3)
        )

    def forward(self, img):
        return self.model(img)    



  
