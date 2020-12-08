from .bases import *


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
class Classifier(BaseModel):
    def __init__(self, model_params):
        super().__init__()
        
        self.attr_from_dict(model_params)
        self.backbone = models.__dict__[self.backbone_type](pretrained=self.pretrained)
        fc_in_channels = self.backbone.fc.in_features
        self.backbone.fc = Identity()
        
        self.fc = nn.Linear(fc_in_channels, self.n_classes)            
                    
        if self.freeze_backbone:
            self.freeze_submodel(self.backbone)   
           
    def forward(self, x):
            
        x = self.backbone(x)
        x = self.fc(x)
        
        return x