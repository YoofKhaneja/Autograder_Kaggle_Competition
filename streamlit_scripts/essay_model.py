import torch
from transformers import AutoModel,AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl


class EssayModel(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.save_hyperparameters()
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.lr = self.config["lr"]
        self.weight_decay = self.config["weight_decay"]
        self.loss_fn = nn.MSELoss()
        
        if self.config["backbone"]=="bert":
            self.bert = AutoModel.from_pretrained(config["backbone_path"])
            self.bert_tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
            self.fc_size=768
        
            
        for param in self.bert.parameters():
            param.requires_grad=False
            
        
        self.pool = nn.AvgPool1d(5,2)
        self.l1 = nn.Linear(self.fc_size,500)
        self.l2 =  nn.Linear(500,200)
        self.l3 = nn.Linear(200,6)
        self.init_weights(self.l3)
    
    def init_weights(self,module):
        if isinstance(module,nn.Linear):
            module.bias.data.fill_(3.5)
            print("Weight initialized")
        
    def forward(self,inputs):
            
        
        X_tokens = self.bert_tokenizer(list(inputs),padding=True,truncation=True,return_tensors="pt")
        X_tokens.to(self.dev)
        x = self.bert(**X_tokens)
        hidden_states = x[0]
        attention_mask = X_tokens["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        temp = torch.sum(hidden_states*mask,1)
        
        x = F.relu(self.l1(temp))
        #print(x.shape)
        x = F.relu(self.l2(x))
        out = self.l3(x)
        return out
    
    def training_step(self,batch,batch_idx):
        X,y = batch
        
        y.to(self.dev)
        out = self(X)
        loss = self.loss_fn(out,y)
        
            
        return loss
    
    def validation_step(self,batch,batch_idx):
        X,y = batch
        
        out = self(X)
        
        loss = self.loss_fn(out,y)
        
        out = out.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        metric = self.MCRMSE(out,y)
        
        return {"metric":metric,"loss":loss}
    
    def MCRMSE(preds,true):
        mse = np.square(preds-true)
        mean_col = np.mean(mse,axis=0)
        final_mean = np.mean(mean_col)
        return final_mean
    
    def training_epoch_end(self,outputs):
        mean_loss = np.mean(np.array([t['loss'].detach().cpu().numpy() for t in outputs]))
        self.log("Train epoch loss: ",mean_loss)
        
        
        
    def validation_epoch_end(self,outputs):
        
        mean_metric = np.mean(np.array([t["metric"] for t in outputs]))
        mean_loss = np.mean(np.array([t["loss"].detach().cpu().numpy() for t in outputs]))
        self.log("val_error",mean_metric)
        self.log("Val epoch loss: ",mean_loss)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
    
def load_model(config):
    cp_model = EssayModel.load_from_checkpoint(config["checkpoint_path"],config=config)
    return cp_model

