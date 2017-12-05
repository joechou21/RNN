import numpy as np
from sklearn.model_selection import train_test_split
import random
import sys
import torch
import torch.nn as nn
import pickle
import gc
from torch.autograd import Variable

np.random.seed(1337)

MAX_LEN = 5
EMBEDDING_SIZE = 5
BATCH_SIZE = 32
EPOCH = 40
DATA_SIZE = 10
INPUT_SIZE = 3



def genTestData(jsonlist,allitem):
 #   print(allitem)
    tensor = []
    print(jsonlist) 
    #tensory= np.zeros(len(jsonlist))
    tensory= []
   
    for idx,user_h in enumerate(jsonlist):
        if len(user_h)>30:
           tensory.append(allitem.index(user_h[30]))
           user_h=user_h[0:30]
           lista=[]
           for idx2, h in enumerate(user_h):
               lista.append(allitem.index(h))
           tensor.append(lista)
          
    #    else:
    #       tensory[idx]=allitem.index(user_h[len(user_h)-1])
    #       lista=[]
          
    #       for i in range(0,30):
    #           if i < len(user_h)-1:
    #                lista.append(allitem.index(user_h[i]))
    #           else:
    #                lista.insert(0,0)
    #       tensor.append(lista)
    return tensor , tensory



def genData(jsonlist,allitem):
   
  
    tensory=np.zeros(16)
    temp=range(1,len(jsonlist))
   
    if len(jsonlist)>16:
        result=random.sample(temp,16)
    #else:
    #   result=random.sample(temp,len(jsonlist)-1)
        tensor=[]
        n=0
        for idx, num in enumerate(result):
            if len(jsonlist[0:num])>30:
               lista=[]
               for i in jsonlist[num-30:num]:
                   lista.append(allitem.index(i))
               tensor.append(lista)
               #print(type(jsonlist[num]))
               
               #print(allitem.index(jsonlist[num]))
               tensory[n]=allitem.index(jsonlist[num])        
               n+=1
        return tensor,tensory[:n]
    else:
        return [],[]

          #  else:
          #     lista=[]
          #     for i in range(0,30):
          #         if i < num:
          #              lista.append(allitem.index(jsonlist[i]))
          #         else:
          #              lista.insert(0,0)
               
          #     tensor.append(lista)
      
def allitem(jsonlist,jsonlist2,jsonlist3):
    itemset=set()
    for userlist in jsonlist:
        for ind in userlist:
            itemset.add(ind)
    for userlist in jsonlist2:
        for ind in userlist:
            itemset.add(ind)
    for userlist in jsonlist3:
        for ind in userlist:
            itemset.add(ind)
    itemlist=list(itemset)
       
    return itemlist
class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, X_list, y_list):
        """
        train one epoch
        """
        y=0
        loss_list = []
        acc_list = []
        num=0
        for X,y in zip(X_list,y_list):
            #print(X)
            #print(y)
            num+=1
           
            if num%1000==0:
                print(num)
            self.optimizer.zero_grad()
            if len(X)==0:
               continue
            if len(X)<16:
               y=y[:len(X)] 
            X=Variable(torch.from_numpy(np.array(X)).transpose(0,1).cuda())   
            output,hn = self.model(X, self.model.initHidden(X.size()[1])) 
          
        #    print("output******",output)
        #    print("Variable**********",Variable(torch.from_numpy(np.array(y)).long().cuda()))
            loss = self.loss_f(output, Variable(torch.from_numpy(np.array(y)).long().cuda()))
            #loss = self.loss_f(output, Variable(torch.from_numpy(np.array(y)).long()))
           
            loss.backward()
            self.optimizer.step()
        ## for log
            loss_list.append(loss.data[0])
            classes = torch.topk(output, 1)[1].data.cpu().numpy().flatten()
            acc = self._accuracy(classes, np.array(y))
            acc_list.append(acc)
        
       
    
        return sum(loss_list) / len(loss_list), sum(acc_list) / len(acc_list)

    def fit(self, X, y, nb_epoch=40, validation_data=()):
        current_loss=0
        print_every=1
        plot_every=1
       
        for t in range(1, nb_epoch + 1):
            loss, acc = self._fit(X, y)
            val_log = ''
      
            if validation_data:
                val_loss, val_acc = self.evaluate(validation_data[0], validation_data[1])
                val_log = "- val_loss: %06.4f - val_acc: %06.4f" % (val_loss, val_acc)
            print("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))

    
    def evaluate(self, X_list, y_list):
        num=0
        for X,y in zip(X_list,y_list):               
                        
            if len(X)==0:
               continue
          #  if len(X)<16:
          #     y=y[:len(X)]
            num+=1
            X=Variable(torch.from_numpy(np.array(X)).transpose(0,1).cuda())
            y_pred, hidden = self.model(X, self.model.initHidden(X.size()[1]))
            loss = self.loss_f(y_pred, Variable(torch.from_numpy(y).long().cuda(),requires_grad=False))
            classes = torch.topk(y_pred, 1)[1].data.cpu().numpy().flatten() 
            #print('classes',classes)
            #print('y',y)
            #print('classes type',type(classes))
            acc = self._accuracy(classes, np.array(y))
        return loss.data[0], acc
    
    
    def predict(self, X, y):
        
        #print(y)
        X=Variable(torch.from_numpy(np.array(X)).transpose(0,1).cuda())
        #print(X)
      
        y_pred, hidden = self.model(X, self.model.initHidden(X.size()[1]))
      
        y_v = Variable(torch.from_numpy(np.array(y)).long().cuda(), requires_grad=False)
        #loss = self.loss_f(y_pred, Variable(torch.from_numpy(np.array(y)).long().cuda()))
        loss = self.loss_f(y_pred, y_v)
        classes = torch.topk(y_pred, 1)[1].data.cpu().numpy().flatten()
        #print(classes)
        #print(y)
        acc = self._accuracy(classes, np.array(y))
        return loss.data[0], acc

    def _accuracy(self, y_pred, y):
        #print('y_pred',y_pred)
        #print('y',y)
        return sum(y_pred == y) / y.shape[0]



class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()

        self.embeds=nn.Embedding(output_size,output_size)
        self.embeds.weight.data.copy_(torch.eye(output_size))

        self.embeds.weight.requires_grad=False
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(30*50, output_size)

    def forward(self, input, hidden):
        #print("input",input) 
        emb=self.embeds(input)
        #print('emb',emb)
       # output, hn = self.gru(input, hidden)
        output, hn = self.gru(emb.cuda(), hidden)
        out= self.fc(output.view(-1,30*50))
        
        return out,hn

    def initHidden(self, N):
        return Variable(torch.zeros(1, N, self.hidden_size).cuda())

def main():
   
   
  
  
 #   emb = embeds(Variable(torch.LongTensor([[1,2,3,4],[2,3,4,5]])))
 #   print(emb)
 #   sys.exit()
    
    #json=[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[1,2,3,4,5,6,7,8,9,10]]
    #json=[[1,2,3,4,5],[11,12,13,14,15,16]]
    #X_train
    #json2=[[1,2],[1,2,3,4,5,6],[11,12,13,14]]
    # with open('train.pkl','rb') as infile:
    #     traindata=pickle.load(infile)
    #with open('val.pkl','rb') as infile:
    #     valdata=pickle.load(infile)
    #with open('test.pkl','rb') as infile:
    #     testdata=pickle.load(infile)
    #print(traindata)
    #print(valdata.index('0'))
    #print(testdata.index('1')

    #itemlist=allitem(json,json,json2)
    #print('itemlist',itemlist) 
    #itemlist=allitem(traindata,valdata,testdata)
   # with open('allitem.pkl','wb') as outfile:
   #      pickle.dump(itemlist,outfile)
    with open('allitem.pkl','rb') as infile:
         itemlist=pickle.load(infile)

   # itemlist.insert(0,'-1')
   
    #print(itemlist)
    #X_test,y_test=genTestData(testdata,itemlist)
    #print('x**************',len(X_test))
    #print('x**************',X_test)
    #print('y**************',len(y_test))
    #print('y*************************',y_test)
 #   X_test,y_test=genTestData(json2,itemlist)
   
    #with open('X_test.pkl','wb') as outfile:
    #     pickle.dump(X_test,outfile)
    #with open('y_test.pkl','wb') as outfile:
    #     pickle.dump(y_test,outfile)
   
   # X_train=[]
   # y_train=[]
   # X_val=[]
   # y_val=[]
   # for i in traindata:
 #   for i in json:
   #     tensorx,tensory=genData(i,itemlist)
   #     X_train.append(tensorx)
   #     y_train.append(tensory)
    #    print(tensorx)
    #    print(tensory)
    #    X=tensorx
    #    print("X",X)
    #    X=np.array(X)
    #    print("X",X)
    #    X=torch.from_numpy(X)
    #    print("X",X)
   
   #     X=X.transpose(0,1)
        #X=Variable(X).cuda()

    #    embeds=nn.Embedding(len(itemlist),len(itemlist))
    #    embeds.weight.data.copy_(torch.from_numpy(np.eye(len(itemlist))))
    #    x=embeds(X)   
    #    print('X.size',x.size()[0])
 #   #    sys.exit()
   # with open('X_train.pkl','wb') as outfile:
   #      pickle.dump(X_train,outfile)
   # with open('y_train.pkl','wb') as outfile:
   #      pickle.dump(y_train,outfile)
   # for i in valdata:
 #   for i in json:
   #     tensorx,tensory=genData(i,itemlist)
   #     X_val.append(tensorx)
   #     y_val.append(tensory)
   # with open('X_val.pkl','wb') as outfile:
   #      pickle.dump(X_val,outfile)
   # with open('y_val.pkl','wb') as outfile:
   #      pickle.dump(y_val,outfile)

        
    with open('X_train.pkl','rb') as infile:
         X_train=pickle.load(infile)
    with open('y_train.pkl','rb') as infile:
         y_train=pickle.load(infile)
    with open('X_val.pkl','rb') as infile:
         X_val=pickle.load(infile)
    with open('y_val.pkl','rb') as infile:
         y_val=pickle.load(infile)
    with open('X_test.pkl','rb') as infile:
         X_test=pickle.load(infile)
    with open('y_test.pkl','rb') as infile:
         y_test=pickle.load(infile)
    #print(X_train[0:1]) 
    #print(y_train[0:1])
    print(len(X_train)) 
    print(len(X_val))
    #sys.exit()
    model = GRU(len(itemlist)+1, 50, len(itemlist)+1).cuda()
   
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adagrad(filter(lambda p:p.requires_grad, model.parameters()), lr=0.1),
        loss=nn.CrossEntropyLoss())
    clf.fit(X_train[0:20000], y_train[0:20000], nb_epoch=EPOCH, validation_data=(X_val[0:1000], y_val[0:1000]))
    score, acc = clf.predict(X_test, y_test)
    print('Test score:', score)
    print('Test accuracy:', acc)

    torch.save(model, 'model.pt')


if __name__ == '__main__':
    main()
